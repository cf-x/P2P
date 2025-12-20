import argparse
import gc
import os
from copy import deepcopy
from functools import partial

import torch
import pandas as pd
from datasets import load_dataset

from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint, save_lora
from hyper_llm_modulator.utils import get_layers
from hyper_llm_modulator.utils.eval_hypermod import gen_and_save_lora
from hyper_llm_modulator.utils.preprocessing import get_preprocessing_fn

from hyper_llm_modulator.vllm_eval import DS_KWARGS, DS_PATHS, TASK_TEMPLATES, eval
from fishfarm.tasks.evalplus import load_dataset as load_evalplus_dataset


def preprocess_result(result, perf_keys):
    keys = set(result.aggregate_metrics.keys()).intersection(perf_keys)
    assert len(keys) == 1
    k = keys.pop()
    return result.aggregate_metrics[k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    perf_keys = set(
        [
            "acc",
            "mbpp_base_pass@1",
            "humaneval_base_pass@1",
            "rouge1_fmeasure",
            "rougeL_fmeasure",
        ]
    )
    device = "cuda"

    results = dict()
    avg_results = dict()
    # load base model and hyperdecoder checkpoint
    (
        model_args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    ) = load_hypermod_checkpoint(args.checkpoint_path, device)

    chat_template = tokenizer.chat_template
    layer_indices = torch.tensor(
        range(len(get_layers(model))), dtype=torch.long, device=device
    )
    save_dir = os.path.dirname(args.checkpoint_path)

    # generate loras
    _gen_and_save_lora = partial(
        gen_and_save_lora,
        model_dir=model_args.model_dir,
        device=device,
        layer_indices=layer_indices,
        emb_model=emb_model,
        emb_tokenizer=emb_tokenizer,
        task_desc_format_fn=task_desc_format_fn,
        pooling_fn=pooling_fn,
        hypermod=hypermod,
    )

    for ds_name in args.tasks:
        # load dataset
        ds_kwargs = None
        if ds_name in ["mbpp", "humaneval"]:
            ds = pd.DataFrame(load_evalplus_dataset(ds_name))
        else:
            ds_path = DS_PATHS[ds_name]
            ds_kwargs = DS_KWARGS[ds_name]
            ds = load_dataset(ds_path, **ds_kwargs)
            preprocessing_fn = get_preprocessing_fn(ds_name)
            ds = ds.map(preprocessing_fn, batched=False).to_pandas()

        template = TASK_TEMPLATES[ds_name] if ds_name in TASK_TEMPLATES else ""
        lora_dir_template = "{save_dir}/generated_loras/{ds_name}/lora_{i}"
        lora_dirs = []
        print(f"Generating LORAs for {ds_name}")
        for i, (idx, sample) in enumerate(ds.iterrows()):
            # generate lora for each sample
            if template:
                txt = template.format(**sample)
            else:
                # mbpp and humaneval
                txt = sample["instruction"]
            lora_dir = lora_dir_template.format(save_dir=save_dir, ds_name=ds_name, i=i)
            lora_dirs.append(lora_dir)
            _gen_and_save_lora(lora_dir=lora_dir, task_desc=txt)

        gc.collect()
        torch.cuda.empty_cache()
        full_results = eval(
            model_args.model_dir,
            lora_dirs,
            ds_name,
            chat_template,
            gpu_memory_utilization=0.6,
            ds_kwargs=ds_kwargs,
            per_sample_lora=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

        results[ds_name] = []
        for lora_dir, res in full_results.items():
            sampled_res_details = res.sample_details
            results[ds_name].append(preprocess_result(res, perf_keys))

        avg_results[ds_name] = sum(results[ds_name]) / len(results[ds_name])

    for ds_name in args.tasks:
        print(f"Average accuracy of {ds_name}: {avg_results[ds_name]}")

    pd.DataFrame({k: [v] for k, v in avg_results.items()}).to_csv(
        f"{save_dir}/avg_eval_results.csv"
    )
