import gc
import os
import json
import hashlib
import logging
import random
import shutil
import statistics
import time
from copy import deepcopy
from functools import partial
from glob import glob

import datasets
import torch
import pandas as pd
import wandb
from tqdm import tqdm

from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint, save_lora
from hyper_llm_modulator.res_aggregator import aggregrate_results_and_save_to_file
from hyper_llm_modulator.utils import generate_simplex_points, get_layers, get_metadata, save_json, log_scalar
from hyper_llm_modulator.data import BENCHMARK_TASK_INFO, check_datasets_for_profile_text, get_profile_text_embs_dict
from hyper_llm_modulator.utils.lora_formatting import convert_qkv_gate_up_lora_to_splits_vllm
from hyper_llm_modulator.utils.model_loading import get_tokenizer
from hyper_llm_modulator.utils.preprocessing import preprocess_result
from hyper_llm_modulator.utils.utils import embed_texts
from hyper_llm_modulator.vllm_eval import eval

logger = logging.getLogger()


CLASSIFICATION_METRICS = {"acc", "f1"}
GENERATION_METRICS = {
    "rouge1_fmeasure",
    "rougeL_fmeasure",
    "meteor",
}
RATING_METRICS = {"mae", "rmse"}
SPECIALTY_METRICS = {"mbpp_base_pass@1", "humaneval_base_pass@1", "stsb"}

ALLOWED_AGG_METRICS = (
    CLASSIFICATION_METRICS
    | GENERATION_METRICS
    | RATING_METRICS
    | SPECIALTY_METRICS
)

RANDOM_PROFILE_PREFIX = "random_profile"


def _derive_random_profile_placeholder(dataset_name: str, formatted_profile: str) -> str:
    seed = f"{dataset_name}:{formatted_profile}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{RANDOM_PROFILE_PREFIX}_{digest[:16]}"


def _load_training_time_seconds(save_dir: str) -> float | None:
    """Best-effort lookup for persisted training duration in the run directory tree."""
    candidate_dirs: list[str] = []
    cur_dir = os.path.abspath(save_dir)
    for _ in range(3):  # search up to grandparent
        if cur_dir in candidate_dirs:
            break
        candidate_dirs.append(cur_dir)
        parent = os.path.dirname(cur_dir)
        if not parent or parent == cur_dir:
            break
        cur_dir = parent

    for directory in candidate_dirs:
        timing_path = os.path.join(directory, "timing_stats.json")
        if os.path.exists(timing_path):
            try:
                with open(timing_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh) or {}
                value = payload.get("total_runtime_seconds")
                if value is None:
                    value = payload.get("training_time_seconds")
                if value is not None:
                    return float(value)
            except Exception as exc:
                logger.warning(f"Failed to read training timing info from {timing_path}: {exc}")
    return None


def _prepare_timing_metadata(
    save_dir: str,
    timing_info: dict | None,
) -> tuple[dict, dict[str, float], float | None, float | None]:
    """Normalize timing payloads and derive aggregate statistics for JSON serialization."""

    timing_info = timing_info or {}
    training_time_seconds = _load_training_time_seconds(save_dir)

    raw_per_lora = timing_info.get("per_lora_seconds", {})
    per_lora_times: dict[str, float] = {}
    if isinstance(raw_per_lora, dict):
        for key, value in raw_per_lora.items():
            if value is None:
                continue
            try:
                per_lora_times[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    per_lora_times = dict(sorted(per_lora_times.items()))

    raw_generation = timing_info.get("per_lora_generation_seconds", {})
    per_lora_generation_times: dict[str, float] = {}
    if isinstance(raw_generation, dict):
        for key, value in raw_generation.items():
            if value is None:
                continue
            try:
                per_lora_generation_times[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    per_lora_generation_times = dict(sorted(per_lora_generation_times.items()))

    raw_per_call = timing_info.get("per_call_seconds", [])
    per_call_timings: list[dict[str, float | None]] = []
    if isinstance(raw_per_call, list):
        for entry in raw_per_call:
            if not isinstance(entry, dict):
                continue
            seconds_val = entry.get("seconds")
            try:
                seconds = float(seconds_val) if seconds_val is not None else None
            except (TypeError, ValueError):
                seconds = None
            per_call_timings.append(
                {
                    "lora_dir": entry.get("lora_dir"),
                    "seconds": seconds,
                }
            )

    total_inference_seconds = timing_info.get("total_inference_seconds")
    if total_inference_seconds is None:
        total_inference_seconds = sum((item["seconds"] or 0.0) for item in per_call_timings)
    try:
        total_inference_seconds = float(total_inference_seconds)
    except (TypeError, ValueError):
        total_inference_seconds = None

    wall_clock_seconds = timing_info.get("wall_clock_seconds")
    try:
        wall_clock_seconds = float(wall_clock_seconds) if wall_clock_seconds is not None else None
    except (TypeError, ValueError):
        wall_clock_seconds = None

    num_calls = timing_info.get("num_calls")
    if num_calls is None:
        num_calls = len(per_call_timings)
    else:
        try:
            num_calls = int(num_calls)
        except (TypeError, ValueError):
            num_calls = len(per_call_timings)

    started_at = timing_info.get("started_at")
    completed_at = timing_info.get("completed_at")

    per_lora_values = list(per_lora_times.values())
    overall_stats: dict[str, float | None] = {
        "num_loras": len(per_lora_values),
        "avg_inference_seconds": None,
        "max_inference_seconds": None,
        "min_inference_seconds": None,
        "median_inference_seconds": None,
        "std_inference_seconds": None,
        "total_inference_seconds": None,
    }
    if per_lora_values:
        avg_val = float(sum(per_lora_values) / len(per_lora_values))
        overall_stats.update(
            {
                "avg_inference_seconds": avg_val,
                "max_inference_seconds": float(max(per_lora_values)),
                "min_inference_seconds": float(min(per_lora_values)),
                "median_inference_seconds": float(statistics.median(per_lora_values)),
                "total_inference_seconds": float(sum(per_lora_values)),
            }
        )
        overall_stats["std_inference_seconds"] = (
            float(statistics.pstdev(per_lora_values)) if len(per_lora_values) > 1 else 0.0
        )

    generation_values = list(per_lora_generation_times.values())
    generation_stats: dict[str, float | None] = {
        "num_loras": len(generation_values),
        "avg_generation_seconds": None,
        "max_generation_seconds": None,
        "min_generation_seconds": None,
        "median_generation_seconds": None,
        "std_generation_seconds": None,
        "total_generation_seconds": None,
    }
    if generation_values:
        avg_generation = float(sum(generation_values) / len(generation_values))
        generation_stats.update(
            {
                "avg_generation_seconds": avg_generation,
                "max_generation_seconds": float(max(generation_values)),
                "min_generation_seconds": float(min(generation_values)),
                "median_generation_seconds": float(statistics.median(generation_values)),
                "total_generation_seconds": float(sum(generation_values)),
            }
        )
        generation_stats["std_generation_seconds"] = (
            float(statistics.pstdev(generation_values))
            if len(generation_values) > 1
            else 0.0
        )

    timing_summary = {
        "training_time_seconds": training_time_seconds,
        "total_inference_seconds": total_inference_seconds,
        "wall_clock_seconds": wall_clock_seconds,
        "num_calls": num_calls,
        "per_lora_inference_seconds": per_lora_times,
        "per_call_inference_seconds": per_call_timings,
        "per_lora_generation_seconds": per_lora_generation_times,
        "started_at": started_at,
        "completed_at": completed_at,
        "overall_statistics": overall_stats,
        "overall_generation_statistics": generation_stats,
    }

    if generation_stats["total_generation_seconds"] is not None:
        timing_summary["total_generation_seconds"] = generation_stats[
            "total_generation_seconds"
        ]

    return timing_summary, per_lora_times, training_time_seconds, total_inference_seconds


def eval_hypermod_checkpoint(
    checkpoint_path,
    device,
    curstep,
    full_eval,
    use_icl=False,
    tasks=None,
    random_profile_embs: bool = False,
    random_profile_strings: bool = False,
    max_context_length: int | None = None,
    rope_scaling: dict | None = None,
    retrieval_k: int | None = None,
    results_save_dir: str | None = None,
):
    # load checkpoint
    args, hypermod, model, tokenizer, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn = (
        load_hypermod_checkpoint(checkpoint_path, device)
    )
    chat_template = tokenizer.chat_template

    run_dir = os.path.dirname(checkpoint_path)
    save_dir = results_save_dir or run_dir
    if results_save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        args_src = os.path.join(run_dir, "args.yaml")
        args_dst = os.path.join(save_dir, "args.yaml")
        if os.path.isfile(args_src) and not os.path.isfile(args_dst):
            try:
                shutil.copy(args_src, args_dst)
            except Exception as exc:
                logger.warning(f"Failed to copy args.yaml to {save_dir}: {exc}")
    train_metadata = get_metadata(args.train_ds_names, args.use_per_task_emb)
    val_metadata = get_metadata(args.eval_ds_info, args.use_per_task_emb)

    if max_context_length is None:
        max_context_length = (
            getattr(args, "max_context_length", None)
            or getattr(args, "max_eval_context_length", None)
            or 2**12
        )

    if retrieval_k is None:
        retrieval_k = getattr(args, "retrieval_k", 4)

    if rope_scaling is None and getattr(args, "use_yarn", False):
        rope_scaling = {
            "type": "yarn",
            "factor": getattr(args, "yarn_scaling_factor", 4.0),
            "original_max_position_embeddings": getattr(
                args, "yarn_original_max_position_embeddings", 32768
            ),
        }

    # Convert eval_ds_info from list to dict format for compatibility
    if isinstance(args.eval_ds_info, list):
        eval_ds_names = args.eval_ds_info
        eval_ds_info = {ds_name: val_metadata.get(ds_name, {}) for ds_name in eval_ds_names}
    else:
        eval_ds_info = deepcopy(val_metadata)
        eval_ds_names = list(eval_ds_info.keys())

    # Only fall back to benchmark defaults when no datasets were configured
    if full_eval:
        for task_name, bench_info in BENCHMARK_TASK_INFO.items():
            if task_name in eval_ds_info:
                ds_kwargs = eval_ds_info[task_name].get("ds_kwargs")
                if not ds_kwargs:
                    eval_ds_info[task_name]["ds_kwargs"] = deepcopy(bench_info)
        eval_ds_names = list(eval_ds_info.keys())

    # Filter by specific tasks if provided
    if tasks is not None:
        logger.info(f"Filtering to specific tasks: {tasks}")
        original_eval_ds_info = deepcopy(eval_ds_info)
        eval_ds_info = {k: v for k, v in eval_ds_info.items() if k in tasks}
        eval_ds_names = list(eval_ds_info.keys())
        # Add any missing tasks that are in benchmark tasks
        for task in tasks:
            if task not in eval_ds_info and task in BENCHMARK_TASK_INFO:
                eval_ds_info[task] = {"ds_kwargs": BENCHMARK_TASK_INFO[task]}
                eval_ds_names.append(task)
        
        # Warn about tasks that couldn't be found
        missing_tasks = [task for task in tasks if task not in eval_ds_names]
        if missing_tasks:
            available_tasks = list(original_eval_ds_info.keys()) + list(BENCHMARK_TASK_INFO.keys())
            logger.warning(f"Tasks not found: {missing_tasks}. Available tasks: {sorted(set(available_tasks))}")
        
        if not eval_ds_names:
            raise ValueError(f"No valid tasks found in the specified list: {tasks}. Available tasks: {sorted(set(list(original_eval_ds_info.keys()) + list(BENCHMARK_TASK_INFO.keys())))}")
    
    # Remove lol_ prefixed datasets
    for ds in list(eval_ds_info.keys()):
        if ds.startswith("lol_"):
            eval_ds_info.pop(ds)
    
    eval_ds_names = list(eval_ds_info.keys())

    # Check if datasets use profile_text mode
    use_profile_text = check_datasets_for_profile_text(eval_ds_names, val_metadata)

    eye = None
    mix_emb = None
    ds_embs_dict = None
    ds_descs = None
    if not args.use_one_hot_task_emb and not use_profile_text:
        ds_descs = {}
        for meta in (train_metadata, val_metadata):
            for ds_name, info in meta.items():
                ds_descs.setdefault(ds_name, info.get("descriptions", []))
    
    # Load peft config once for constructing VLLM instances
    base_hypermod_dir = os.path.dirname(checkpoint_path)
    if "checkpoint" in base_hypermod_dir:
        base_hypermod_dir = base_hypermod_dir.split("checkpoint")[0]

    from peft import get_peft_config, PeftConfig

    peft_config = get_peft_config(
        PeftConfig.from_json_file(f"{base_hypermod_dir}/adapter_config.json")
    )

    llm_common_kwargs = dict(
        model=args.model_dir,
        seed=42,
        max_model_len=max_context_length,
        enable_lora=True,
        max_lora_rank=peft_config.r,
        gpu_memory_utilization=0.6,
    )
    if rope_scaling is not None:
        llm_common_kwargs["rope_scaling"] = rope_scaling

    logger.info(
        "Sequentially generating personalized LoRAs and evaluating %d datasets",
        len(eval_ds_names),
    )

    import vllm
    from hyper_llm_modulator.vllm_eval import LoRAVLLMModel

    for idx, eval_ds in enumerate(eval_ds_names, start=1):
        logger.info("(%d/%d) Generating LoRAs for %s", idx, len(eval_ds_names), eval_ds)

        if hypermod is None:
            (
                args,
                hypermod,
                model,
                tokenizer,
                emb_model,
                emb_tokenizer,
                task_desc_format_fn,
                pooling_fn,
            ) = load_hypermod_checkpoint(checkpoint_path, device)
            if chat_template is None:
                chat_template = tokenizer.chat_template

        layer_indices = torch.tensor(
            range(len(get_layers(model))), dtype=torch.long, device=device
        )

        ds_kwargs = None
        if eval_ds in eval_ds_info and "ds_kwargs" in eval_ds_info[eval_ds]:
            ds_kwargs = (
                eval_ds_info[eval_ds]["ds_kwargs"]
                if eval_ds_info[eval_ds]["ds_kwargs"]
                else None
            )

        if args.use_one_hot_task_emb:
            eye = torch.eye(len(args.train_ds_names)).to(device)
            mix_emb = generate_simplex_points(
                n_points=3, dimension=len(args.train_ds_names)
            ).to(device)
            lora_dirs, task_save_dicts = generate_loras_for_task_one_hot(
                args=args,
                hypermod=hypermod,
                layer_indices=layer_indices,
                save_dir=save_dir,
                device=device,
                eval_task=eval_ds,
                eye=eye,
                mix_emb=mix_emb,
            )
        else:
            _gen_and_save_lora = partial(
                gen_and_save_lora,
                model_dir=args.model_dir,
                device=device,
                layer_indices=layer_indices,
                emb_model=emb_model,
                emb_tokenizer=emb_tokenizer,
                task_desc_format_fn=task_desc_format_fn,
                pooling_fn=pooling_fn,
                hypermod=hypermod,
            )

            if use_profile_text:
                if random_profile_embs:
                    logger.info(
                        "Random profile embeddings enabled for %s (ablation mode)",
                        eval_ds,
                    )
                ds_embs_dict = get_profile_text_embs_dict(
                    args,
                    emb_model,
                    emb_tokenizer,
                    task_desc_format_fn,
                    pooling_fn,
                    [eval_ds],
                    val_metadata,
                    device,
                )
                lora_dirs, task_save_dicts = generate_loras_for_task_from_profile_text(
                    model_dir=args.model_dir,
                    eval_task=eval_ds,
                    val_metadata=val_metadata,
                    save_dir=save_dir,
                    device=device,
                    layer_indices=layer_indices,
                    emb_model=emb_model,
                    emb_tokenizer=emb_tokenizer,
                    task_desc_format_fn=task_desc_format_fn,
                    pooling_fn=pooling_fn,
                    hypermod=hypermod,
                    args=args,
                    emb_cache=ds_embs_dict,
                    gen_and_save_lora_fn=_gen_and_save_lora,
                    use_random_profile_embs=random_profile_embs,
                    use_random_profile_strings=random_profile_strings,
                )
                if random_profile_strings and task_save_dicts:
                    ordered_random_profiles: list[str | None] = [None] * len(task_save_dicts)
                    valid_random_profiles = True
                    for payload in task_save_dicts:
                        idx = payload.get("sample_idx")
                        placeholder = payload.get("random_profile_string")
                        if idx is None or placeholder is None:
                            valid_random_profiles = False
                            break
                        if idx >= len(ordered_random_profiles):
                            ordered_random_profiles.extend([None] * (idx - len(ordered_random_profiles) + 1))
                        ordered_random_profiles[idx] = placeholder

                    if valid_random_profiles and all(ordered_random_profiles):
                        new_dataset_path = _rewrite_dataset_with_random_profiles(
                            eval_ds,
                            ds_kwargs,
                            [str(p) for p in ordered_random_profiles],
                            save_dir,
                        )
                        if new_dataset_path:
                            existing_kwargs = ds_kwargs if isinstance(ds_kwargs, dict) else {}
                            eval_ds_info.setdefault(eval_ds, {}).setdefault(
                                "ds_kwargs", dict(existing_kwargs)
                            )
                            eval_ds_info[eval_ds]["ds_kwargs"]["path"] = new_dataset_path
                            ds_kwargs = eval_ds_info[eval_ds]["ds_kwargs"]
                    else:
                        logger.warning(
                            "Random profile strings enabled for %s but placeholder metadata was incomplete; skipping dataset rewrite",
                            eval_ds,
                        )
            else:
                eval_info = eval_ds_info.get(eval_ds, {})
                lora_dirs, task_save_dicts = generate_loras_for_task_from_descs(
                    eval_task=eval_ds,
                    eval_info=eval_info,
                    random_descs=args.additional_eval_descs,
                    train_metadata=train_metadata,
                    val_metadata=val_metadata,
                    ds_descs=ds_descs,
                    save_dir=save_dir,
                    gen_and_save_lora_fn=_gen_and_save_lora,
                )

        if not lora_dirs:
            logger.warning(
                "No LoRAs generated for %s; skipping personalized evaluation", eval_ds
            )
            hypermod = None
            model = None
            tokenizer = None
            emb_model = None
            emb_tokenizer = None
            task_desc_format_fn = None
            pooling_fn = None
            layer_indices = None
            torch.cuda.empty_cache()
            gc.collect()
            continue

        if use_profile_text and task_save_dicts:
            try:
                first_save_dict = task_save_dicts[0]
                split_type = first_save_dict.get("split")
                if split_type in [
                    "per_sample_profile",
                    "per_user_profile",
                    "per_profile_text",
                ]:
                    logger.info(
                        "Validating %s LoRA setup for %s...", split_type, eval_ds
                    )
                    is_valid = validate_profile_lora_setup(
                        eval_ds,
                        lora_dirs,
                        task_save_dicts,
                        val_metadata,
                    )
                    if not is_valid:
                        logger.error(
                            "Validation failed for %s, skipping evaluation", eval_ds
                        )
                        hypermod = None
                        model = None
                        tokenizer = None
                        emb_model = None
                        emb_tokenizer = None
                        task_desc_format_fn = None
                        pooling_fn = None
                        layer_indices = None
                        torch.cuda.empty_cache()
                        gc.collect()
                        task_dir = os.path.join(save_dir, "generated_loras", eval_ds)
                        if os.path.exists(task_dir):
                            shutil.rmtree(task_dir)
                        continue
            except Exception as exc:
                logger.warning(
                    "Validation failed for %s with error: %s. Proceeding.", eval_ds, exc
                )

        # Release generation models before loading the inference model
        tokenizer = None
        emb_model = None
        emb_tokenizer = None
        task_desc_format_fn = None
        pooling_fn = None
        hypermod = None
        model = None
        layer_indices = None
        eye = None
        mix_emb = None
        ds_embs_dict = None
        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Running personalized evaluation for %s", eval_ds)

        llm_kwargs = dict(llm_common_kwargs)

        vllm_model = vllm.LLM(**llm_kwargs)
        sampling_params = vllm.SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=2**9,
            repetition_penalty=1.0,
        )
        model_wrapper = LoRAVLLMModel(
            llm=vllm_model,
            sampling_params=sampling_params,
            chat_template=chat_template,
            prefill_text="",
        )

        results = do_eval_task_with_shared_model(
            model_wrapper,
            args.model_dir,
            chat_template,
            save_dir,
            lora_dirs,
            eval_ds,
            task_save_dicts,
            ds_kwargs,
            use_icl,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
            retrieval_k=retrieval_k,
        )

        del vllm_model, model_wrapper, sampling_params
        torch.cuda.empty_cache()
        gc.collect()

        task_dir = os.path.join(save_dir, "generated_loras", eval_ds)
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
            torch.cuda.empty_cache()

    
    # aggregate eval results (only for primary run directory to ensure metadata is present)
    df = None
    if results_save_dir is None:
        df = aggregrate_results_and_save_to_file(
            base_model_dir=args.model_dir,
            mt_lora_dir=args.mt_lora_path,
            hypermod_dir=save_dir,
            hypermod_name="hyperlora",
        )
    else:
        logger.debug(
            "Skipping aggregate results for alternate evaluation directory %s; using root stats instead",
            save_dir,
        )
    out = {}
    if df is not None and isinstance(df, pd.DataFrame) and "benchmark_avg" in df:
        indexer = df["benchmark_avg"]

        def add_metric(metric_key, wandb_key):
            try:
                value = indexer.loc[metric_key]
            except KeyError:
                logger.debug(
                    "Skipping benchmark metric %s for hyperlora; entry not found in aggregated results",
                    metric_key,
                )
                return
            out[wandb_key] = value

        prefix = "test" if full_eval else "val"
        add_metric(("hyperlora", "other_train_descs"), f"{prefix}/benchmark/acc/other_train_descs")
        add_metric(("hyperlora", "random_descs"), f"{prefix}/benchmark/acc/random_descs")
        add_metric(("hyperlora", "eval_descs"), f"{prefix}/benchmark/acc/eval_descs")
        add_metric(("hyperlora", "train_descs"), f"{prefix}/benchmark/acc/train_descs")

    if wandb.run is not None and out:
        wandb.log(out, step=curstep)
    return out


def eval_lora(
    args,
    lora_dir,
    curstep,
    full_eval=False,
    use_icl=False,
    max_context_length: int | None = None,
    rope_scaling: dict | None = None,
    retrieval_k: int | None = None,
):
    save_dicts = None
    all_lora_dirs = [lora_dir]
    chat_template = get_tokenizer(args.model_dir).chat_template

    # Convert eval_ds_info from list to dict format for compatibility  
    if isinstance(args.eval_ds_info, list):
        eval_ds_names = args.eval_ds_info
        val_metadata = get_metadata(args.eval_ds_info, args.use_per_task_emb)
        eval_ds_info = {ds_name: val_metadata.get(ds_name, {}) for ds_name in eval_ds_names}
    else:
        eval_ds_info = deepcopy(args.eval_ds_info)
        eval_ds_names = list(eval_ds_info.keys())

    if full_eval:
        eval_ds_info = {k: v for k, v in eval_ds_info.items() if k in BENCHMARK_TASK_INFO}
        eval_ds_names = list(eval_ds_info.keys())

    for k in eval_ds_names:
        if k in BENCHMARK_TASK_INFO:
            eval_ds_info.setdefault(k, {})
            eval_ds_info[k]["ds_kwargs"] = BENCHMARK_TASK_INFO[k]

    if max_context_length is None:
        max_context_length = (
            getattr(args, "max_context_length", None)
            or getattr(args, "max_eval_context_length", None)
            or 2**12
        )

    if retrieval_k is None:
        retrieval_k = getattr(args, "retrieval_k", 4)

    if rope_scaling is None and getattr(args, "use_yarn", False):
        rope_scaling = {
            "type": "yarn",
            "factor": getattr(args, "yarn_scaling_factor", 4.0),
            "original_max_position_embeddings": getattr(
                args, "yarn_original_max_position_embeddings", 32768
            ),
        }

    # Create a single VLLM model instance to reuse across all evaluations
    logger.info("Creating single VLLM model instance for LoRA evaluations...")
    
    import vllm
    from hyper_llm_modulator.vllm_eval import LoRAVLLMModel
    
    # Determine appropriate LoRA rank for evaluation
    max_lora_rank = 64
    adapter_config_path = os.path.join(lora_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as fh:
                adapter_cfg = json.load(fh)
            max_lora_rank = int(adapter_cfg.get("r", max_lora_rank))
        except Exception as exc:
            logger.warning(f"Failed to read adapter config at {adapter_config_path}: {exc}")

    # Create the VLLM model instance
    llm_kwargs = dict(
        model=args.model_dir,
        seed=42,
        max_model_len=max_context_length,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=0.6,
    )
    if rope_scaling is not None:
        llm_kwargs["rope_scaling"] = rope_scaling

    vllm_model = vllm.LLM(**llm_kwargs)
    
    sampling_params = vllm.SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=2**9,
        repetition_penalty=1.0,
    )
    
    model_wrapper = LoRAVLLMModel(
        llm=vllm_model,
        sampling_params=sampling_params,
        chat_template=chat_template,
        prefill_text=""
    )

    logger.info(f"Evaluating LoRA on {len(eval_ds_names)} datasets...")
    for eval_ds in tqdm(eval_ds_names, desc="Evaluating LoRA on datasets", unit="dataset"):
        ds_kwargs = eval_ds_info[eval_ds]["ds_kwargs"] if eval_ds in eval_ds_info and "ds_kwargs" in eval_ds_info[eval_ds] else None
        do_eval_task_with_shared_model(
            model_wrapper,
            args.model_dir,
            chat_template,
            lora_dir,
            all_lora_dirs,
            eval_ds,
            save_dicts,
            ds_kwargs,
            use_icl,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
            retrieval_k=retrieval_k,
        )

    # Clean up the VLLM model
    del vllm_model, model_wrapper, sampling_params
    gc.collect()
    torch.cuda.empty_cache()

    perf_files = glob(f"{lora_dir}/eval_results/*_eval_results.json")
    perf_files = [f for f in perf_files if not f.startswith("lol")]

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    for perf_file in perf_files:
        with open(perf_file, "r") as f:
            perf_dict = json.load(f)

        for dataset_entries in perf_dict.values():
            if not isinstance(dataset_entries, list):
                continue

            for entry in dataset_entries:
                if not isinstance(entry, dict):
                    continue
                results_dict = entry.get("results", {})
                if not isinstance(results_dict, dict):
                    continue

                for metric_name, metric_value in results_dict.items():
                    if (
                        metric_name in ALLOWED_AGG_METRICS
                        and isinstance(metric_value, (int, float))
                    ):
                        metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(metric_value)
                        metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

    aggregated_metrics = {
        metric: metric_sums[metric] / metric_counts[metric]
        for metric in metric_sums
        if metric_counts[metric] > 0
    }

    if not aggregated_metrics:
        logger.warning("No valid evaluation metrics found; returning empty aggregation")
        aggregated_metrics = {"acc": 0.0}

    rows = [
        {
            "model_name": "mt_lora",
            "split": "test" if full_eval else "val",
            "metric": metric,
            "value": value,
        }
        for metric, value in sorted(aggregated_metrics.items())
    ]
    df = pd.DataFrame(rows)
    df.to_csv(f"{lora_dir}/eval_results/combined_results.csv", index=False)

    prefix = "test" if full_eval else "val"
    for metric, value in aggregated_metrics.items():
        log_scalar(f"{prefix}/benchmark/{metric}/avg", value, curstep)

    return aggregated_metrics


@torch.no_grad()
def generate_lora_for_tasks_one_hot(
    args,
    hypermod,
    layer_indices,
    save_dir,
    device,
):
    if isinstance(args.eval_ds_info, list):
        eval_ds_names = args.eval_ds_info
    else:
        eval_ds_names = list(args.eval_ds_info.keys())

    all_lora_dirs = {eval_task: [] for eval_task in eval_ds_names}
    save_dicts = {eval_task: [] for eval_task in eval_ds_names}

    eye = torch.eye(len(args.train_ds_names)).to(device)
    mix_emb = generate_simplex_points(n_points=3, dimension=len(args.train_ds_names)).to(device)

    valid_eval_tasks = [task for task in eval_ds_names if task in args.train_ds_names]
    logger.info(f"Generating LoRAs for {len(valid_eval_tasks)} tasks...")

    for eval_task in tqdm(valid_eval_tasks, desc="Generating LoRAs (one-hot)", unit="task"):
        lora_dirs, metadata = generate_loras_for_task_one_hot(
            args=args,
            hypermod=hypermod,
            layer_indices=layer_indices,
            save_dir=save_dir,
            device=device,
            eval_task=eval_task,
            eye=eye,
            mix_emb=mix_emb,
        )
        all_lora_dirs[eval_task].extend(lora_dirs)
        save_dicts[eval_task].extend(metadata)

    return all_lora_dirs, save_dicts


@torch.no_grad()
def generate_loras_for_task_one_hot(
    args,
    hypermod,
    layer_indices,
    save_dir,
    device,
    eval_task,
    eye,
    mix_emb,
):
    if eval_task not in args.train_ds_names:
        logger.warning(f"Task {eval_task} not found in training datasets; skipping one-hot LoRA generation")
        return [], []

    splits = ["train_descs", "other_train_descs", "random_descs"]
    train_idx = args.train_ds_names.index(eval_task)
    train_emb = eye[train_idx].unsqueeze(0)
    non_train_idx = [i for i in range(len(args.train_ds_names)) if i != train_idx]
    random_train_emb = None
    if non_train_idx:
        random_train_idx = random.choice(non_train_idx)
        random_train_emb = eye[random_train_idx].unsqueeze(0)

    all_embs = [train_emb, random_train_emb, mix_emb]
    valid_splits_embs = [(split, embs) for split, embs in zip(splits, all_embs) if embs is not None]

    lora_dirs: list[str] = []
    metadata: list[dict] = []

    for split, embs in tqdm(
        valid_splits_embs,
        desc=f"LoRA splits for {eval_task}",
        leave=False,
    ):
        dirs = [f"{save_dir}/generated_loras/{eval_task}/{split}/lora_{i}" for i in range(len(embs))]
        for lora_dir, task_emb in tqdm(
            list(zip(dirs, embs)),
            desc=f"Generating {split} LoRAs for {eval_task}",
            leave=False,
        ):
            metadata.append({
                "task_emb": task_emb.cpu().numpy().tolist(),
                "split": split,
                "lora_dir": lora_dir,
            })
            lora_dirs.append(lora_dir)

            task_emb = task_emb.unsqueeze(0)
            encoded_task_emb = hypermod.task_encoder(task_emb)["encoded_task_emb"].detach()
            lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb)
            save_lora(lora_sd, hypermod.peft_config, lora_dir)

    return lora_dirs, metadata


@torch.no_grad()
def generate_loras_for_tasks_from_descs(
    model_dir,
    eval_ds_info,
    random_descs,
    train_metadata,
    val_metadata,
    save_dir,
    device,
    layer_indices,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    hypermod,
):
    _gen_and_save_lora = partial(
        gen_and_save_lora,
        model_dir=model_dir,
        device=device,
        layer_indices=layer_indices,
        emb_model=emb_model,
        emb_tokenizer=emb_tokenizer,
        task_desc_format_fn=task_desc_format_fn,
        pooling_fn=pooling_fn,
        hypermod=hypermod,
    )

    ds_descs = {ds: train_metadata[ds]["descriptions"] for ds in train_metadata}
    ds_descs.update({ds: val_metadata[ds]["descriptions"] for ds in val_metadata})

    all_lora_dirs = {eval_task: [] for eval_task in eval_ds_info}
    save_dicts = {eval_task: [] for eval_task in eval_ds_info}

    logger.info(f"Generating LoRAs for {len(eval_ds_info)} tasks from descriptions...")
    for eval_task, eval_info in tqdm(
        eval_ds_info.items(), desc="Generating LoRAs (descriptions)", unit="task"
    ):
        lora_dirs, metadata = generate_loras_for_task_from_descs(
            eval_task=eval_task,
            eval_info=eval_info,
            random_descs=random_descs,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            ds_descs=ds_descs,
            save_dir=save_dir,
            gen_and_save_lora_fn=_gen_and_save_lora,
        )
        all_lora_dirs[eval_task].extend(lora_dirs)
        save_dicts[eval_task].extend(metadata)
    return all_lora_dirs, save_dicts


@torch.no_grad()
def generate_loras_for_task_from_descs(
    eval_task: str,
    eval_info: dict,
    random_descs,
    train_metadata,
    val_metadata,
    ds_descs,
    save_dir: str,
    gen_and_save_lora_fn,
):
    random_descs = random_descs or []
    splits = ["train_descs", "eval_descs", "other_train_descs", "random_descs"]

    eval_descs = eval_info.get("descriptions", []) or []
    train_descs = []
    if eval_task in train_metadata and train_metadata[eval_task].get("descriptions"):
        train_descs = train_metadata[eval_task]["descriptions"][0:1]

    other_train_descs = []
    for ds_name, descs in ds_descs.items():
        if ds_name == eval_task:
            continue
        if descs:
            other_train_descs.append(random.choice(descs))
    if other_train_descs:
        other_train_descs = random.sample(other_train_descs, k=min(len(other_train_descs), 3))

    all_split_descs = list(zip(splits, [train_descs, eval_descs, other_train_descs, random_descs]))

    lora_dirs: list[str] = []
    metadata: list[dict] = []

    for split, descs in tqdm(
        all_split_descs,
        desc=f"LoRA splits for {eval_task}",
        leave=False,
    ):
        if not descs:
            continue
        dirs = [f"{save_dir}/generated_loras/{eval_task}/{split}/lora_{i}" for i in range(len(descs))]
        for lora_dir, task_desc in tqdm(
            list(zip(dirs, descs)),
            desc=f"Generating {split} LoRAs for {eval_task}",
            leave=False,
        ):
            generation_seconds = gen_and_save_lora_fn(
                lora_dir=lora_dir, task_desc=task_desc
            )
            metadata.append(
                {
                    "task_desc": task_desc,
                    "split": split,
                    "lora_dir": lora_dir,
                    "lora_generation_seconds": generation_seconds,
                }
            )
            lora_dirs.append(lora_dir)

    return lora_dirs, metadata


@torch.no_grad()
def gen_and_save_lora(
    model_dir,
    device,
    layer_indices,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    hypermod,
    lora_dir,
    task_desc,
):
    task_emb = embed_texts([task_desc], emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device)
    encoder_out = hypermod.task_encoder(task_emb)
    encoded_task_emb = encoder_out["encoded_task_emb"].detach()
    gen_start = time.perf_counter()
    lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb)
    generation_seconds = time.perf_counter() - gen_start
    save_lora(lora_sd, hypermod.peft_config, lora_dir)
    hypermod.model_config.save_pretrained(lora_dir)
    if "Phi-3" in model_dir:
        convert_qkv_gate_up_lora_to_splits_vllm(lora_dir)
    return float(generation_seconds)


@torch.no_grad()
def generate_loras_for_tasks_from_profile_text(
    model_dir,
    eval_ds_names,
    val_metadata,
    save_dir,
    device,
    layer_indices,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    hypermod,
    args,
    ds_embs_dict=None,
    use_random_profile_embs: bool = False,
    use_random_profile_strings: bool = False,
):
    """Generate LoRAs using user_id mapping to ensure samples from the same user share LoRAs."""
    
    _gen_and_save_lora = partial(
        gen_and_save_lora,
        model_dir=model_dir,
        device=device,
        layer_indices=layer_indices,
        emb_model=emb_model,
        emb_tokenizer=emb_tokenizer,
        task_desc_format_fn=task_desc_format_fn,
        pooling_fn=pooling_fn,
        hypermod=hypermod,
    )

    # Get profile_text embeddings for all datasets if not provided
    if ds_embs_dict is None:
        ds_embs_dict = get_profile_text_embs_dict(
            args, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn,
            eval_ds_names, val_metadata, device
        )

    all_lora_dirs = {eval_task: [] for eval_task in eval_ds_names}
    save_dicts = {eval_task: [] for eval_task in eval_ds_names}
    
    logger.info(f"Generating LoRAs for {len(eval_ds_names)} tasks using user_id mapping...")
    for eval_task in tqdm(
        eval_ds_names,
        desc="Generating LoRAs (profile text)",
        unit="task",
    ):
        lora_dirs, metadata = generate_loras_for_task_from_profile_text(
            model_dir=model_dir,
            eval_task=eval_task,
            val_metadata=val_metadata,
            save_dir=save_dir,
            device=device,
            layer_indices=layer_indices,
            emb_model=emb_model,
            emb_tokenizer=emb_tokenizer,
            task_desc_format_fn=task_desc_format_fn,
            pooling_fn=pooling_fn,
            hypermod=hypermod,
            args=args,
            emb_cache=ds_embs_dict,
            gen_and_save_lora_fn=_gen_and_save_lora,
            use_random_profile_embs=use_random_profile_embs,
            use_random_profile_strings=use_random_profile_strings,
        )
        all_lora_dirs[eval_task].extend(lora_dirs)
        save_dicts[eval_task].extend(metadata)

    return all_lora_dirs, save_dicts


@torch.no_grad()
def generate_loras_for_task_from_profile_text(
    model_dir,
    eval_task,
    val_metadata,
    save_dir,
    device,
    layer_indices,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    hypermod,
    args,
    emb_cache,
    gen_and_save_lora_fn,
    use_random_profile_embs: bool = False,
    use_random_profile_strings: bool = False,
):
    from hyper_llm_modulator.data import load_and_format_dataset

    lora_dirs: list[str] = []
    metadata: list[dict] = []

    if eval_task not in emb_cache:
        logger.warning(f"No embeddings found for {eval_task}, skipping")
        return lora_dirs, metadata

    emb_data = emb_cache[eval_task]

    if isinstance(emb_data, dict) and "profile_lists" in emb_data:
        formatted_dataset = load_and_format_dataset(
            val_metadata,
            emb_tokenizer,
            args.sft_mode,
            is_intx_model=emb_tokenizer.chat_template is not None,
            ds_name=eval_task,
            ds_kwargs=val_metadata[eval_task]["ds_kwargs"],
        )

        profile_lists = emb_data["profile_lists"]
        profile_to_emb = emb_data["profile_to_emb"]
        formatted_profiles = emb_data.get("formatted_profiles")

        random_profile_choices = (
            list(profile_to_emb.items())
            if use_random_profile_embs and not use_random_profile_strings
            else None
        )
        if use_random_profile_embs and not use_random_profile_strings and not random_profile_choices:
            logger.warning(
                "Random profile embeddings requested for %s but no cached embeddings were found; using true profiles.",
                eval_task,
            )

        if formatted_profiles is None:
            formatted_profiles = []
            from hyper_llm_modulator.utils.preprocessing import format_profile_text
            user_profile_format = emb_data.get("user_profile_format", getattr(args, "user_profile_format", "history"))
            profile_k = emb_data.get("profile_k", getattr(args, "profile_k", 0))
            include_history_stat = emb_data.get("include_history_stat", getattr(args, "include_history_stat", False))
            for profile_text_str, profile_all_history_str, data_entry in profile_lists:
                formatted_profiles.append(
                    format_profile_text(
                        profile_text_str,
                        user_profile_format,
                        profile_all_history_str,
                        data_entry,
                        profile_k,
                        eval_task,
                        include_history_stat,
                    )
                )

        if len(formatted_profiles) != len(profile_lists):
            logger.error(
                f"Formatted profile list length mismatch for {eval_task}: "
                f"{len(formatted_profiles)} vs {len(profile_lists)}"
            )
            return lora_dirs, metadata

        user_ids = formatted_dataset["user_id"] if "user_id" in formatted_dataset.column_names else [None] * len(profile_lists)
        profile_text_raw = (
            formatted_dataset["profile_text"]
            if "profile_text" in formatted_dataset.column_names
            else [None] * len(profile_lists)
        )

        profile_to_samples: dict[str, list[int]] = {}
        for idx, formatted_profile in enumerate(formatted_profiles):
            profile_to_samples.setdefault(formatted_profile, []).append(idx)

        logger.info(
            f"Dataset {eval_task}: {len(profile_lists)} samples grouped into {len(profile_to_samples)} unique formatted profiles"
        )

        placeholder_map: dict[str, str] = {}
        placeholder_emb_cache: dict[str, torch.Tensor] = {}
        profile_random_embedding_source: dict[str, str | None] = {}
        profile_random_embedding_applied: dict[str, bool] = {}
        if use_random_profile_strings:
            placeholder_map = {
                formatted_profile: _derive_random_profile_placeholder(
                    eval_task, str(formatted_profile)
                )
                for formatted_profile in profile_to_samples.keys()
            }
            logger.info(
                "Random profile strings enabled for %s; replacing %d unique profiles with placeholders",
                eval_task,
                len(placeholder_map),
            )

        profile_to_lora_dir: dict[str, str] = {}
        profile_generation_seconds: dict[str, float] = {}
        for formatted_profile, sample_indices in tqdm(
            profile_to_samples.items(),
            total=len(profile_to_samples),
            desc=f"Generating profile LoRAs for {eval_task}",
            leave=False,
        ):
            profile_hash = hashlib.sha256(formatted_profile.encode("utf-8")).hexdigest()[:16]
            lora_dir = f"{save_dir}/generated_loras/{eval_task}/per_profile_text/profile_{profile_hash}"
            profile_to_lora_dir[formatted_profile] = lora_dir

            random_source_profile = None
            placeholder_text = (
                placeholder_map.get(formatted_profile)
                if use_random_profile_strings
                else None
            )

            random_embedding_applied = False

            if use_random_profile_strings and placeholder_text is not None:
                cached_emb = placeholder_emb_cache.get(placeholder_text)
                if cached_emb is None:
                    generated_emb = embed_texts(
                        [placeholder_text],
                        emb_model,
                        emb_tokenizer,
                        lambda x: x,
                        pooling_fn,
                        device,
                    )
                    cached_emb = generated_emb.squeeze(0).detach().cpu()
                    placeholder_emb_cache[placeholder_text] = cached_emb
                task_emb = placeholder_emb_cache[placeholder_text].unsqueeze(0).to(device)
            elif use_random_profile_embs and random_profile_choices:
                random_source_profile, random_emb_tensor = random.choice(
                    random_profile_choices
                )
                task_emb = random_emb_tensor.unsqueeze(0).to(device)
                random_embedding_applied = True
            elif formatted_profile in profile_to_emb:
                task_emb = profile_to_emb[formatted_profile].unsqueeze(0).to(device)
            else:
                logger.warning(
                    f"Formatted profile not found in cached embeddings; regenerating for dataset {eval_task}"
                )
                task_emb = embed_texts(
                    [formatted_profile],
                    emb_model,
                    emb_tokenizer,
                    lambda x: x,
                    pooling_fn,
                    device,
                )

            profile_random_embedding_source[formatted_profile] = random_source_profile
            profile_random_embedding_applied[formatted_profile] = random_embedding_applied

            encoder_out = hypermod.task_encoder(task_emb)
            encoded_task_emb = encoder_out["encoded_task_emb"].detach()
            gen_start = time.perf_counter()
            lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb)
            profile_generation_seconds[formatted_profile] = float(
                time.perf_counter() - gen_start
            )

            os.makedirs(lora_dir, exist_ok=True)
            try:
                save_lora(lora_sd, hypermod.peft_config, lora_dir)
                hypermod.model_config.save_pretrained(lora_dir)
                if "Phi-3" in model_dir:
                    convert_qkv_gate_up_lora_to_splits_vllm(lora_dir)
            except Exception as exc:
                logger.error(f"Failed to save LoRA for formatted profile hash {profile_hash}: {exc}")
                if os.path.exists(lora_dir):
                    shutil.rmtree(lora_dir)
                raise

        for sample_idx in range(len(profile_lists)):
            formatted_profile = formatted_profiles[sample_idx]
            placeholder_text = (
                placeholder_map.get(formatted_profile)
                if use_random_profile_strings
                else None
            )
            lora_dir = profile_to_lora_dir[formatted_profile]
            raw_profile = profile_text_raw[sample_idx]
            save_payload = {
                "sample_idx": sample_idx,
                "formatted_profile_text": placeholder_text
                if placeholder_text is not None
                else formatted_profile,
                "split": "per_profile_text",
                "lora_dir": lora_dir,
                "shared_with_samples": profile_to_samples[formatted_profile],
            }
            generation_seconds = profile_generation_seconds.get(formatted_profile)
            if generation_seconds is not None:
                save_payload["lora_generation_seconds"] = generation_seconds
            if user_ids[sample_idx] is not None:
                save_payload["user_id"] = user_ids[sample_idx]
            if raw_profile is not None:
                save_payload["profile_text"] = raw_profile
            if profile_random_embedding_applied.get(formatted_profile):
                save_payload["random_profile_embedding"] = True
                source_profile = profile_random_embedding_source.get(formatted_profile)
                if source_profile is not None:
                    save_payload["random_profile_embedding_source"] = source_profile
            if use_random_profile_strings and placeholder_text is not None:
                save_payload["random_profile_string"] = placeholder_text
                save_payload["original_formatted_profile_text"] = formatted_profile

            metadata.append(save_payload)
            lora_dirs.append(lora_dir)

        logger.info(
            f"Generated {len(profile_to_lora_dir)} LoRAs for {eval_task} covering {len(profile_lists)} samples"
        )

        return lora_dirs, metadata

    logger.warning(f"Profile text not available for {eval_task}, using fallback descriptions")
    if eval_task in val_metadata and "descriptions" in val_metadata[eval_task]:
        descs = val_metadata[eval_task]["descriptions"][:3]
        for i, desc in enumerate(descs):
            lora_dir = f"{save_dir}/generated_loras/{eval_task}/fallback_desc/lora_{i}"
            generation_seconds = gen_and_save_lora_fn(
                lora_dir=lora_dir, task_desc=desc
            )
            metadata.append(
                {
                    "task_desc": desc,
                    "split": "fallback_desc",
                    "lora_dir": lora_dir,
                    "lora_generation_seconds": generation_seconds,
                }
            )
            lora_dirs.append(lora_dir)

    return lora_dirs, metadata


def _rewrite_dataset_with_random_profiles(
    task_name: str,
    ds_kwargs: dict | None,
    random_profiles: list[str],
    save_dir: str,
) -> str | None:
    if not ds_kwargs:
        logger.warning(
            "Random profile strings requested for %s but dataset kwargs were empty; skipping dataset rewrite",
            task_name,
        )
        return None

    dataset_path = ds_kwargs.get("path") if isinstance(ds_kwargs, dict) else None
    if not dataset_path or not os.path.exists(dataset_path):
        logger.warning(
            "Random profile strings requested for %s but dataset path %s was unavailable; skipping dataset rewrite",
            task_name,
            dataset_path,
        )
        return None

    try:
        dataset_obj = datasets.load_from_disk(dataset_path)
    except Exception as exc:
        logger.warning(
            "Failed to load dataset for %s from %s while preparing random profiles: %s",
            task_name,
            dataset_path,
            exc,
        )
        return None

    target_key = None
    if isinstance(dataset_obj, datasets.DatasetDict):
        split_key = ds_kwargs.get("split") if isinstance(ds_kwargs, dict) else None
        name_key = ds_kwargs.get("name") if isinstance(ds_kwargs, dict) else None
        if split_key and split_key in dataset_obj:
            target_key = split_key
        elif name_key and name_key in dataset_obj:
            target_key = name_key
        else:
            target_key = next(iter(dataset_obj.keys()))
        dataset_to_randomize = dataset_obj[target_key]
    else:
        dataset_to_randomize = dataset_obj

    if len(dataset_to_randomize) != len(random_profiles):
        logger.warning(
            "Random profile string count (%d) did not match dataset size (%d) for %s; skipping dataset rewrite",
            len(random_profiles),
            len(dataset_to_randomize),
            task_name,
        )
        return None

    update_columns: list[str] = []
    for column in dataset_to_randomize.column_names:
        if column in {"profile_text", "profile_all_history"} or column.startswith("profile_retrieval_k"):
            update_columns.append(column)

    if not update_columns:
        logger.info(
            "Dataset %s did not contain profile-related columns to rewrite; skipping random profile injection",
            task_name,
        )
        return None

    def _inject_random_profile(example: dict, idx: int):
        placeholder = random_profiles[idx]
        for column in update_columns:
            example[column] = placeholder
        return example

    randomized_dataset = dataset_to_randomize.map(
        _inject_random_profile,
        with_indices=True,
        load_from_cache_file=False,
        desc=f"Injecting random profiles for {task_name}",
    )

    randomized_dir = os.path.join(save_dir, "random_profile_datasets", task_name)
    os.makedirs(os.path.dirname(randomized_dir), exist_ok=True)
    if os.path.exists(randomized_dir):
        shutil.rmtree(randomized_dir)

    if isinstance(dataset_obj, datasets.DatasetDict):
        datasets.DatasetDict({target_key: randomized_dataset}).save_to_disk(randomized_dir)
    else:
        randomized_dataset.save_to_disk(randomized_dir)

    logger.info("Saved random-profile dataset for %s to %s", task_name, randomized_dir)
    return randomized_dir


@torch.no_grad()
def validate_profile_lora_setup(eval_dataset, lora_dirs, save_dicts, val_metadata):
    """Validate that the per-profile LoRA setup is correct."""
    try:
        from hyper_llm_modulator.data import load_dataset_with_local_support

        ds_kwargs = val_metadata[eval_dataset]["ds_kwargs"]
        dataset = load_dataset_with_local_support(**ds_kwargs)
        if isinstance(dataset, dict):
            dataset = list(dataset.values())[0]

        num_samples = len(dataset)
        num_lora_entries = len(lora_dirs)

        logger.info(
            f"Validation: Dataset {eval_dataset} has {num_samples} samples; received {num_lora_entries} LoRA assignments"
        )

        if num_lora_entries != num_samples:
            logger.error(
                f"MISMATCH: Expected {num_samples} LoRA entries but received {num_lora_entries} for dataset {eval_dataset}"
            )
            return False

        if save_dicts and len(save_dicts) != num_lora_entries:
            logger.error(
                f"MISMATCH: save_dict count ({len(save_dicts)}) != LoRA entries ({num_lora_entries})"
            )
            return False

        if save_dicts:
            formatted_to_dir: dict[str, str] = {}
            for idx, (save_dict, lora_dir) in enumerate(zip(save_dicts, lora_dirs)):
                sample_idx = save_dict.get("sample_idx")
                if sample_idx != idx:
                    logger.error(
                        f"Sample index mismatch at position {idx}: expected {idx}, got {sample_idx}"
                    )
                    return False

                formatted_profile = save_dict.get("formatted_profile_text")
                if formatted_profile is not None:
                    existing_dir = formatted_to_dir.get(formatted_profile)
                    if existing_dir is None:
                        formatted_to_dir[formatted_profile] = lora_dir
                    elif existing_dir != lora_dir:
                        logger.error(
                            "Inconsistent LoRA assignment: formatted profile reused with different LoRA directories"
                        )
                        return False

        logger.info(f"✓ Validation passed: {eval_dataset} per-profile LoRA configuration looks consistent")
        return True

    except Exception as exc:
        logger.error(f"Validation failed for {eval_dataset}: {exc}")
        return False


@torch.no_grad()
def do_eval_task_with_shared_model(
    model_wrapper,
    model_dir: str,
    chat_template: str | None,
    save_dir: str,
    lora_dirs: list[str],
    eval_dataset: str,
    save_dicts: list[dict] = None,
    ds_kwargs: dict = None,
    use_icl: bool = False,
    max_context_length: int | None = None,
    rope_scaling: dict | None = None,
    retrieval_k: int | None = None,
):
    """
    Evaluate a task using a pre-created VLLM model instance.
    This avoids reloading the base model for each task evaluation.
    """
    perf_keys = ALLOWED_AGG_METRICS
    os.makedirs(f"{save_dir}/eval_results", exist_ok=True)
    results = {eval_dataset: []}
    if save_dicts is None:
        save_dicts = [dict() for _ in lora_dirs]
    
    # Determine if this is user_id-based LoRA mode by checking save_dicts
    is_per_sample_lora = False
    if save_dicts and len(save_dicts) > 0:
        # Check if this is profile_text mode with user_id-based LoRAs
        first_save_dict = save_dicts[0]
        split_type = first_save_dict.get("split")
        if isinstance(first_save_dict, dict) and split_type in ["per_sample_profile", "per_user_profile", "per_profile_text"]:
            is_per_sample_lora = True
            if split_type == "per_user_profile":
                unique_lora_dirs = list(set(lora_dirs))
                logger.info(f"Using user_id-based LoRA mode for {eval_dataset} with {len(unique_lora_dirs)} user-specific LoRAs for {len(lora_dirs)} samples")
            elif split_type == "per_profile_text":
                unique_lora_dirs = list(set(lora_dirs))
                logger.info(
                    f"Using per-profile-text LoRA mode for {eval_dataset} with {len(unique_lora_dirs)} unique profiles across {len(lora_dirs)} samples"
                )
            else:
                logger.info(f"Using per-sample LoRA mode for {eval_dataset} with {len(lora_dirs)} sample-specific LoRAs")
    
    # Call the eval function with the pre-created model
    eval_output = eval(
        model_dir,
        lora_dirs,
        eval_dataset,
        chat_template,
        gpu_memory_utilization=0.6,
        ds_kwargs=ds_kwargs,
        use_icl=use_icl,
        per_sample_lora=is_per_sample_lora,  # Enable per-sample LoRA if detected
        pre_created_model=model_wrapper,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
        retrieval_k=retrieval_k,
    )
    full_results = eval_output.get("results", {})
    timing_info = eval_output.get("timing", {})
    generation_timings: dict[str, float] = {}
    if save_dicts:
        for entry in save_dicts:
            if not isinstance(entry, dict):
                continue
            lora_dir = entry.get("lora_dir")
            generation_seconds = entry.get("lora_generation_seconds")
            if (
                lora_dir
                and generation_seconds is not None
                and lora_dir not in generation_timings
            ):
                try:
                    generation_timings[lora_dir] = float(generation_seconds)
                except (TypeError, ValueError):
                    continue
    if generation_timings:
        per_lora_generation = timing_info.setdefault(
            "per_lora_generation_seconds", {}
        )
        if isinstance(per_lora_generation, dict):
            for key, value in generation_timings.items():
                per_lora_generation[key] = value
        else:
            timing_info["per_lora_generation_seconds"] = dict(
                generation_timings
            )
    combined_generation_seconds = (
        sum(generation_timings.values()) if generation_timings else None
    )
    timing_summary, per_lora_times, training_time_seconds, total_inference_seconds = _prepare_timing_metadata(
        save_dir, timing_info
    )

    # Handle results based on whether per-sample LoRA was used
    if is_per_sample_lora and "per_sample_lora_combined" in full_results:
        # For per-sample LoRA, we get a combined result
        combined_res = full_results["per_sample_lora_combined"]
        
        # Determine if this is user_id-based or sample-based
        first_save_dict = save_dicts[0] if save_dicts else {}
        split_type = first_save_dict.get("split", "unknown")
        
        if split_type == "per_user_profile":
            # User_id-based LoRA mode
            unique_lora_dirs = list(set(lora_dirs))
            split_name = "per_user_profile_combined"
            num_loras_desc = f"{len(unique_lora_dirs)} user LoRAs"
        elif split_type == "per_profile_text":
            unique_lora_dirs = list(set(lora_dirs))
            split_name = "per_profile_text_combined"
            num_loras_desc = f"{len(unique_lora_dirs)} profile LoRAs"
        else:
            # Legacy per-sample mode
            split_name = "per_sample_profile_combined"
            num_loras_desc = f"{len(lora_dirs)} sample LoRAs"
        
        # Create a single result entry with combined metrics
        sampled_res_details = combined_res.sample_details
        results[eval_dataset].append(
            dict(
                results=preprocess_result(combined_res, perf_keys),
                sampled_res_details=sampled_res_details,
                split=split_name,
                num_samples=len(combined_res.sample_details),
                num_loras_description=num_loras_desc,
                combined_inference_time_seconds=total_inference_seconds,
                combined_wall_clock_seconds=timing_summary.get("wall_clock_seconds"),
                training_time_seconds=training_time_seconds,
            )
        )
        if combined_generation_seconds is not None:
            results[eval_dataset][-1][
                "combined_generation_time_seconds"
            ] = combined_generation_seconds

        # Also add individual sample information for analysis
        for i, (lora_dir, save_dict) in enumerate(zip(lora_dirs, save_dicts)):
            if i < len(combined_res.sample_details):
                sample_detail = combined_res.sample_details[i]
                lora_time = per_lora_times.get(lora_dir)
                results[eval_dataset].append(
                    dict(
                        results={"individual_sample": sample_detail.get("is_correct", 0)},
                        sample_details=[sample_detail],
                        **save_dict,
                        inference_time_seconds=lora_time,
                        training_time_seconds=training_time_seconds,
                    )
                )
    else:
        # Traditional mode - handle each LoRA result separately
        for (lora_dir, res), save_dict in zip(full_results.items(), save_dicts):
            sampled_res_details = res.sample_details
            lora_time = per_lora_times.get(lora_dir)
            results[eval_dataset].append(
                dict(
                    results=preprocess_result(res, perf_keys),
                    sampled_res_details=sampled_res_details,
                    **save_dict,
                    inference_time_seconds=lora_time,
                    training_time_seconds=training_time_seconds,
                )
            )

    torch.cuda.empty_cache()
    gc.collect()
    result_path = f"{save_dir}/eval_results/{eval_dataset}_eval_results.json"
    results.setdefault("_metadata", {})["timing"] = timing_summary
    save_json(results, result_path)
    return results


@torch.no_grad()
def do_eval_task(
    model_dir: str,
    chat_template: str | None,
    save_dir: str,
    lora_dirs: list[str],
    eval_dataset: str,
    save_dicts: list[dict] = None,
    ds_kwargs: dict = None,
    use_icl: bool = False,
    max_context_length: int | None = None,
    rope_scaling: dict | None = None,
    retrieval_k: int | None = None,
):
    """Original do_eval_task function - kept for backward compatibility"""
    perf_keys = ALLOWED_AGG_METRICS
    os.makedirs(f"{save_dir}/eval_results", exist_ok=True)
    results = {eval_dataset: []}
    if save_dicts is None:
        save_dicts = [dict() for _ in lora_dirs]
    eval_output = eval(
        model_dir,
        lora_dirs,
        eval_dataset,
        chat_template,
        gpu_memory_utilization=0.6,
        ds_kwargs=ds_kwargs,
        use_icl=use_icl,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
        retrieval_k=retrieval_k,
    )
    full_results = eval_output.get("results", {})
    timing_info = eval_output.get("timing", {})
    timing_summary, per_lora_times, training_time_seconds, total_inference_seconds = _prepare_timing_metadata(
        save_dir, timing_info
    )

    for (lora_dir, res), save_dict in zip(full_results.items(), save_dicts):
        sampled_res_details = res.sample_details[:10]
        lora_time = per_lora_times.get(lora_dir)
        results[eval_dataset].append(
            dict(
                results=preprocess_result(res, perf_keys),
                sampled_res_details=sampled_res_details,
                **save_dict,
                inference_time_seconds=lora_time,
                training_time_seconds=training_time_seconds,
            )
        )

    torch.cuda.empty_cache()
    gc.collect()
    result_path = f"{save_dir}/eval_results/{eval_dataset}_eval_results.json"
    results.setdefault("_metadata", {})["timing"] = timing_summary
    save_json(results, result_path)
    return results
