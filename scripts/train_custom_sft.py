from copy import deepcopy
import gc
import os
import random
import string
import subprocess
import time
from glob import glob

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from datasets import disable_caching
from transformers import set_seed, get_scheduler

from hyper_llm_modulator.configs import ArgumentParser, TrainingArguments
from hyper_llm_modulator.data import create_dataloaders
from hyper_llm_modulator.hyper_modulator import create_hypermod
from hyper_llm_modulator.sft_trainer import train
from hyper_llm_modulator.utils import (
    get_layers,
    get_num_params,
    create_logger,
    save_yaml,
    get_model_and_tokenizer,
    get_peft_config,
    get_pooling_fn,
    add_full_stop,
    get_metadata,
    get_tokenizer,
)
from hyper_llm_modulator.utils.model_loading import get_emb_model_and_fns
from hyper_llm_modulator.utils.eval_hypermod import eval_hypermod_checkpoint, eval_lora
import yaml

MODEL_INPUT_KEYS = ["input_ids", "attention_mask"]


def _configure_gradient_checkpointing(model, logger):
    """Enable gradient checkpointing and disable caches across wrapped models."""

    def _set_use_cache_false(module):
        if module is None:
            return
        if hasattr(module, "config") and getattr(module.config, "use_cache", True):
            module.config.use_cache = False
        if hasattr(module, "generation_config") and getattr(module.generation_config, "use_cache", True):
            module.generation_config.use_cache = False

    def _enable_input_grads(module):
        if module is None:
            return
        if hasattr(module, "enable_input_require_grads"):
            try:
                module.enable_input_require_grads()
            except Exception:
                pass

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        # Older Transformers versions do not accept gradient_checkpointing_kwargs
        model.gradient_checkpointing_enable()

    # Disable caches and ensure input grads on the model and common wrappers
    for attr in [None, "model", "base_model", "module"]:
        target = model if attr is None else getattr(model, attr, None)
        _set_use_cache_false(target)
        _enable_input_grads(target)

    logger.info("Gradient checkpointing enabled; caches disabled for wrapped model hierarchy")


def log_num_train_params(model):
    logger.debug("Trainable model parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.debug(f"{name}, dtype:{p.dtype}")

    num_total_params, num_trainable_params = get_num_params(model)
    logger.info(
        f"trainable params: {num_trainable_params:,d} "
        f"|| all params: {num_total_params:,d} "
        f"|| trainable%: {100 * num_trainable_params / num_total_params:.4f}"
    )


def main(args):
    args.train_ds_names = args.train_ds_names[: args.n_train_ds]
    args.use_hypernet = use_hypernet = "hyper" in args.exp_setup
    # get task metadata and save to the corresponding run folder
    train_metadata = get_metadata(args.train_ds_names, args.use_per_task_emb)
    val_metadata = get_metadata(args.eval_ds_info, args.use_per_task_emb)
    
    # Note: Individual task descriptions will be handled per-dataset during data loading
    # instead of creating a combined task description for all tasks
    
    save_dir = args.save_dir
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    save_yaml(vars(args), f"{save_dir}/args.yaml")
    set_seed(args.seed)
    # load peft config
    peft_config = None
    peft_type = args.exp_setup.split("_")[-1]
    if peft_type == "lora":
        # used for both normal training and hypernet training
        # for hypernet, the init weights will be copied as the output bias
        peft_config = get_peft_config(
            args.model_dir,
            peft_type,
            target_modules=args.target_modules,
        )
        peft_config.save_pretrained(save_dir)
        logger.debug(f"peft_config:\n{peft_config}")
    else:
        logger.warning(
            "=" * 60 + f"\npeft_type: {peft_type}. Doing normal full-finetuning without any PEFT.\n" + "=" * 60
        )

    # setup accelerator
    plugin = GradientAccumulationPlugin(num_steps=args.grad_accum_steps, sync_with_dataloader=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_plugin=plugin,
        split_batches=True,  # True means do not multiply batch size by the number of gpus used
        log_with="wandb",
    )

    def clear_mem():
        nonlocal accelerator
        torch.cuda.empty_cache()
        accelerator.free_memory()
        gc.collect()

    wandb_dir = f"{os.environ['HOME']}/.wandb/logs/{os.environ['WANDB_PROJECT']}/"
    os.makedirs(wandb_dir, exist_ok=True)
    accelerator.init_trackers(
        os.getenv("WANDB_PROJECT"),
        config=vars(args),
        init_kwargs=dict(wandb={"group": args.run_name, "name": args.run_name, "dir": wandb_dir, "notes": args.notes}),
    )
    device = accelerator.device

    ##############################################################################
    # emb model setup (for dataloaders)
    ##############################################################################
    emb_model = None
    emb_tokenizer = None
    task_desc_format_fn = None
    use_explicit_emb_model = False
    pooling_fn = None

    # Check if we're using an explicit embedding model
    if use_hypernet and not args.use_one_hot_task_emb and args.emb_model:
        # Case 1: Using explicit embedding model - load it first for dataloaders
        use_explicit_emb_model = True
        emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = get_emb_model_and_fns(
            args.emb_model, device, args.user_profile_format
        )
        logger.debug(f"emb_model: {emb_model}")
        emb_model.eval()
        
        # Load a temporary tokenizer for the main model (needed for dataloaders)
        temp_tokenizer = get_tokenizer(args.model_dir, train=True, peft_config=peft_config)
        
        ##############################################################################
        # Dataset setup (with explicit emb model)
        ##############################################################################
        data_loaders = create_dataloaders(
            args,
            train_metadata,
            val_metadata,
            use_hypernet,
            device,
            temp_tokenizer,
            True,  # is_intx_model - will be validated later with actual model
            emb_model,
            emb_tokenizer,
            task_desc_format_fn,
            pooling_fn,
        )
        train_dataloader = data_loaders["train"]
        val_dataloaders = {k: v for k, v in data_loaders.items() if "train" not in k}

        # Clean up explicit emb model to free memory before loading main model
        del emb_model, emb_tokenizer
        clear_mem()
        logger.info("Cleaned up explicit embedding model to free memory before loading main model")

        ##############################################################################
        # Model setup
        ##############################################################################
        model, tokenizer = get_model_and_tokenizer(
            args.model_dir,
            train=True,
            requires_grad=True if "fullfinetune" in args.exp_setup else False,
            peft_config=peft_config,
            model_kwargs={"output_hidden_states": True, "output_attentions": False},
            device=device,
        )
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            _configure_gradient_checkpointing(model, logger)
        # train to output delta_w for all layers
        layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)
        # NOTE: the module_names is needed for saving generated lora or vera to lora format
        # this is required for using vllm during evaluation
        is_intx_model = tokenizer.chat_template is not None
        assert is_intx_model, "Only chat models are supported"
        logger.debug(f"Model config: {model.config}")
        logger.debug(f"Model: {model}")
        logger.debug(f"is_intx_model: {is_intx_model}")
        logger.debug(f"Tokenizer: {tokenizer}")
        logger.debug(f"layer_indices: {layer_indices}")

        ##############################################################################
        # hypermod setup (reload explicit emb model)
        ##############################################################################
        hypermod = None
        if use_hypernet:
            task_emb_size = None
            if not args.use_one_hot_task_emb:
                # Reload explicit embedding model for training
                emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = get_emb_model_and_fns(
                    args.emb_model, device, args.user_profile_format
                )
                logger.debug(f"emb_model reloaded: {emb_model}")
                emb_model.eval()
                task_emb_size = emb_model.config.hidden_size
            
            hypermod = create_hypermod(args, peft_type, device, model, layer_indices, task_emb_size)
            logger.debug(f"Hypermod: {hypermod}")
            model.add_module("hypermod", hypermod)
        elif "lora" in args.exp_setup:
            # for training vanilla LoRA
            model.set_adapter("default")
        else:
            model.train()

    else:
        # Case 2: Using main model as emb model OR not using hypernet - original order
        ##############################################################################
        # Model setup
        ##############################################################################
        model, tokenizer = get_model_and_tokenizer(
            args.model_dir,
            train=True,
            requires_grad=True if "fullfinetune" in args.exp_setup else False,
            peft_config=peft_config,
            model_kwargs={"output_hidden_states": True, "output_attentions": False},
            device=device,
        )
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            _configure_gradient_checkpointing(model, logger)
        # train to output delta_w for all layers
        layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)
        # NOTE: the module_names is needed for saving generated lora or vera to lora format
        # this is required for using vllm during evaluation
        is_intx_model = tokenizer.chat_template is not None
        assert is_intx_model, "Only chat models are supported"
        logger.debug(f"Model config: {model.config}")
        logger.debug(f"Model: {model}")
        logger.debug(f"is_intx_model: {is_intx_model}")
        logger.debug(f"Tokenizer: {tokenizer}")
        logger.debug(f"layer_indices: {layer_indices}")

        ##############################################################################
        # emb model and hypermod setup (using main model)
        ##############################################################################
        hypermod = None
        if use_hypernet:
            task_emb_size = None
            if not args.use_one_hot_task_emb:
                # Use main model as embedding model
                emb_model = model
                emb_tokenizer = deepcopy(tokenizer)
                task_desc_format_fn = add_full_stop
                pooling_fn = get_pooling_fn("last_token")
                emb_model.eval()
                task_emb_size = emb_model.config.hidden_size
            
            hypermod = create_hypermod(args, peft_type, device, model, layer_indices, task_emb_size)
            logger.debug(f"Hypermod: {hypermod}")
            model.add_module("hypermod", hypermod)
        elif "lora" in args.exp_setup:
            # for training vanilla LoRA
            model.set_adapter("default")
        else:
            model.train()

        ##############################################################################
        # Dataset setup (with main model as emb model)
        ##############################################################################
        data_loaders = create_dataloaders(
            args,
            train_metadata,
            val_metadata,
            use_hypernet,
            device,
            tokenizer,
            is_intx_model,
            emb_model,
            emb_tokenizer,
            task_desc_format_fn,
            pooling_fn,
        )
        train_dataloader = data_loaders["train"]
        val_dataloaders = {k: v for k, v in data_loaders.items() if "train" not in k}

    log_num_train_params(model)

    ##############################################################################
    # Training
    ##############################################################################
    wd = args.weight_decay
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=wd)
    model, hypermod, optimizer = accelerator.prepare(model, hypermod, optimizer)
    train_dataloader = accelerator.prepare(train_dataloader)
    for k, v in val_dataloaders.items():
        val_dataloaders[k] = accelerator.prepare(v)
    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = args.warmup_frac * num_training_steps
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=int(num_warmup_steps / args.grad_accum_steps),
        num_training_steps=int(num_training_steps / args.grad_accum_steps),
    )
    scheduler = accelerator.prepare(scheduler)
    inp_dropout = getattr(peft_config, f"{peft_type.lower()}_dropout", 0.0)

    if use_explicit_emb_model:
        del emb_model, emb_tokenizer
        clear_mem()

    train(
        args,
        save_dir,
        inp_dropout,
        accelerator,
        model,
        layer_indices,
        hypermod,
        train_dataloader,
        val_dataloaders,
        optimizer,
        num_training_steps,
        scheduler,
    )


def evaluate_best_checkpoint_on_failure(args, save_dir, device):
    """Find the existing best (or latest) checkpoint and evaluate it.

    Priority order for hypernet:
    1) best_checkpoint_info.yaml -> best_checkpoint_path
    2) latest checkpoints/it_*/hypermod.pt by largest step
    
    For LoRA:
    1) Latest checkpoints/it_*/ (contains adapter_model.safetensors)
    2) Fallback to save_dir if it contains adapter_model.safetensors
    """
    try:
        use_hypernet = getattr(args, "use_hypernet", None)
        if use_hypernet is None:
            use_hypernet = ("hyper" in args.exp_setup)

        if use_hypernet:
            # Try best_checkpoint_info.yaml
            best_info_path = os.path.join(save_dir, "best_checkpoint_info.yaml")
            checkpoint_path = None
            if os.path.isfile(best_info_path):
                try:
                    with open(best_info_path, "r") as f:
                        best_info = yaml.safe_load(f) or {}
                    candidate = best_info.get("best_checkpoint_path")
                    if candidate and os.path.isfile(candidate):
                        checkpoint_path = candidate
                        logger.info(f"Found best checkpoint from metadata: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to read {best_info_path}: {e}")

            # Fallback: latest checkpoint by numeric iteration
            if checkpoint_path is None:
                cp_files = []
                for d in glob(os.path.join(save_dir, "checkpoints", "it_*")):
                    try:
                        step = int(os.path.basename(d).split("it_")[-1])
                    except Exception:
                        step = -1
                    pt = os.path.join(d, "hypermod.pt")
                    if os.path.isfile(pt):
                        cp_files.append((step, pt))
                if cp_files:
                    cp_files.sort(key=lambda x: x[0])
                    checkpoint_path = cp_files[-1][1]
                    logger.info(f"Using latest checkpoint: {checkpoint_path}")

            if checkpoint_path is None:
                logger.error("No hypermod checkpoint found to evaluate after failure.")
                return

            eval_hypermod_checkpoint(checkpoint_path, device, curstep=0, full_eval=True)
        else:
            # LoRA path
            lora_dirs = []
            for d in glob(os.path.join(save_dir, "checkpoints", "it_*")):
                try:
                    step = int(os.path.basename(d).split("it_")[-1])
                except Exception:
                    step = -1
                adapter = os.path.join(d, "adapter_model.safetensors")
                if os.path.isfile(adapter):
                    lora_dirs.append((step, d))
            if lora_dirs:
                lora_dirs.sort(key=lambda x: x[0])
                lora_dir = lora_dirs[-1][1]
                logger.info(f"Using latest LoRA checkpoint: {lora_dir}")
                eval_lora(args, lora_dir, curstep=0, full_eval=True)
            elif os.path.isfile(os.path.join(save_dir, "adapter_model.safetensors")):
                logger.info(f"Using LoRA artifacts in {save_dir}")
                eval_lora(args, save_dir, curstep=0, full_eval=True)
            else:
                logger.error("No LoRA checkpoint found to evaluate after failure.")
    except Exception as e:
        logger.error(f"Failed to evaluate checkpoint after failure: {e}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_PROJECT"] = "hypermod_sft"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    disable_caching()

    parser = ArgumentParser((TrainingArguments,))
    args = parser.parse()
    assert (
        args.use_per_task_emb + args.use_inp_as_desc + args.use_per_sample_desc + args.use_default_desc
    ) <= 1, "only one or none of use_per_task_emb, use_inp_as_desc, use_per_sample_desc can be used"

    assert (
        args.use_per_task_emb or not args.use_one_hot_task_emb
    ), "one_hot_task_emb can only be used with use_per_task_emb"

    uuid = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
    args.run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    args.save_dir = f"train_outputs/sft/{args.exp_setup}/{args.run_name}"
    global logger
    logger = create_logger(args.save_dir, debug=args.debug)
    logger.debug(f"CMD: {' '.join(os.sys.argv)}")
    logger.debug(f"args: {args}")
    logger.debug(f"Is CUDA available: {torch.cuda.is_available()}")
    logger.debug(f"CUDA Index: {torch.cuda.current_device()}")
    logger.debug(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    try:
        main(args)
    except RuntimeError as e:
        # Fallback path: if training stops due to NaN loss, try to eval best/last checkpoint
        if "nan" in str(e).lower():
            logger.error("Training aborted due to NaN loss. Attempting to evaluate best/last checkpoint...")
            evaluate_best_checkpoint_on_failure(args, args.save_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            raise
    subprocess.run("wandb sync --no-include-online --clean", shell=True)
    subprocess.run("wandb artifact cache cleanup 10GB", shell=True)
