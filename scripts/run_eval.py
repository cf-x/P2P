import argparse
import logging
import gc
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import datasets
import torch

from hyper_llm_modulator.utils import save_json, get_tokenizer, get_metadata_for_task
from hyper_llm_modulator.vllm_eval import eval


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_task_user_metadata(task_name: str) -> Optional[Dict[str, Any]]:
    """Load user-related metadata (user_id, question_id) for a task once."""
    try:
        metadata = get_metadata_for_task(task_name)
    except FileNotFoundError:
        return None
    ds_kwargs = metadata.get("ds_kwargs") or {}
    dataset_path = ds_kwargs.get("path")
    if not dataset_path or not os.path.exists(dataset_path):
        return None

    try:
        dataset = datasets.load_from_disk(dataset_path)
    except Exception as exc:  # pragma: no cover - informational logging only
        logger.warning("Unable to load dataset for task %s: %s", task_name, exc)
        return None

    if isinstance(dataset, datasets.DatasetDict):
        split = ds_kwargs.get("split")
        if split and split in dataset:
            dataset = dataset[split]
        else:
            ds_name = ds_kwargs.get("name")
            if ds_name and ds_name in dataset:
                dataset = dataset[ds_name]
            else:
                first_split = next(iter(dataset.keys()))
                dataset = dataset[first_split]

    if "user_id" not in dataset.column_names:
        return None

    user_ids = list(dataset["user_id"])
    qid_lookup: Dict[str, Any] = {}
    for key_field in ("question_id", "id"):
        if key_field in dataset.column_names:
            keys = list(dataset[key_field])
            qid_lookup = {
                str(key): uid
                for key, uid in zip(keys, user_ids)
                if key is not None and uid is not None
            }
            break

    return {"user_ids": user_ids, "qid_lookup": qid_lookup}


def _ensure_user_ids(task_name: str, sample_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every sample detail has a user_id when available in the source dataset."""
    if not sample_details:
        return sample_details

    if all(detail.get("user_id") not in (None, "") for detail in sample_details):
        return sample_details

    user_metadata = _load_task_user_metadata(task_name)
    if not user_metadata:
        return sample_details

    user_ids: List[Any] = user_metadata.get("user_ids") or []
    qid_lookup: Dict[str, Any] = user_metadata.get("qid_lookup") or {}

    enriched_details: List[Dict[str, Any]] = []
    for idx, detail in enumerate(sample_details):
        enriched = dict(detail)
        current_user_id = enriched.get("user_id")
        if current_user_id not in (None, ""):
            enriched_details.append(enriched)
            continue

        mapped_id = None
        for key_name in ("question_id", "id", "sample_id"):
            key_value = enriched.get(key_name)
            if key_value is None:
                continue
            mapped_id = qid_lookup.get(str(key_value))
            if mapped_id is not None:
                break

        if mapped_id is None and idx < len(user_ids):
            mapped_id = user_ids[idx]

        if mapped_id not in (None, ""):
            enriched["user_id"] = mapped_id

        enriched_details.append(enriched)

    return enriched_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--lora-dirs", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--use-icl", action="store_true")
    parser.add_argument("--use-task-desc", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--save-to-base-model-dir", action="store_true")
    parser.add_argument("--max-context-length", type=int, default=3000, help="Maximum context length for the model (default: 3000)")
    
    # YaRN rope scaling arguments
    parser.add_argument("--use-yarn", action="store_true", help="Enable YaRN rope scaling for context extension")
    parser.add_argument("--yarn-scaling-factor", type=float, default=4.0, help="YaRN scaling factor (default: 4.0)")
    parser.add_argument("--yarn-original-max-position-embeddings", type=int, default=32768, help="Original max position embeddings (default: 32768)")
    
    # RAG retrieval k parameter
    parser.add_argument("--retrieval-k", type=int, choices=[1, 2, 4, 8, 12, 16, 32], default=4, help="Number of retrieved history items to include in RAG prompts (default: 4)")
    
    args = parser.parse_args()
    print(args)
    
    # Build rope_scaling config if yarn is enabled
    rope_scaling = None
    if args.use_yarn:
        rope_scaling = {
            "factor": args.yarn_scaling_factor,
            "original_max_position_embeddings": args.yarn_original_max_position_embeddings,
            "type": "yarn"
        }
        print(f"Using YaRN rope scaling: {rope_scaling}")
    
    tokenizer = get_tokenizer(args.model_dir)
    for task in args.tasks:
        print(f"Evaluating {task}")
        json_name = f"{task}_eval_results"
        if args.use_icl:
            json_name += "_icl"
        if args.use_task_desc:
            json_name += "_task_desc"
        if args.use_yarn:
            json_name += "_yarn"
            
        # Add retrieval_k suffix for RAG and PAG tasks
        if task.startswith("RAG_") or task.startswith("PAG_"):
            json_name += f"_k{args.retrieval_k}"
        eval_output = eval(
            args.model_dir,
            args.lora_dirs,
            task=task,
            chat_template=tokenizer.chat_template,
            use_icl=args.use_icl,
            use_task_desc=args.use_task_desc,
            max_context_length=args.max_context_length,
            rope_scaling=rope_scaling,
            retrieval_k=args.retrieval_k,
        )
        res = eval_output.get("results", {})
        timing_info = eval_output.get("timing", {})
        per_lora_times = timing_info.get("per_lora_seconds", {})
        total_inference_seconds = timing_info.get("total_inference_seconds")
        training_time_seconds = None

        for k in res:
            print(k)
            print(res[k].aggregate_metrics)
            if args.save_results and args.lora_dirs:
                result_path = f"{k}/eval_results/{json_name}.json"
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                # Save ALL sampled details for every task to match hypermod format
                sampled_details = res[k].sample_details
                sampled_details = _ensure_user_ids(task, sampled_details)
                inference_time = per_lora_times.get(k)
                save_json(
                    {
                        task: [
                            dict(
                                results=res[k].aggregate_metrics,
                                sampled_res_details=sampled_details,
                                num_samples=len(sampled_details),
                                path=k,
                                inference_time_seconds=inference_time,
                                training_time_seconds=training_time_seconds,
                            )
                        ]
                    },
                    result_path,
                )

        if args.save_to_base_model_dir:
            if not args.lora_dirs:
                result_path = (
                    f"eval_results/{args.model_dir}/base_model/{json_name}.json"
                )
            else:
                result_path = (
                    f"eval_results/{args.model_dir}/lora/{json_name}_lora.json"
                )
            os.makedirs(os.path.dirname(result_path), exist_ok=True)

            # Include ALL sampled_res_details for every task to match hypermod format
            results_payload = {task: []}
            for k in res:
                sampled_details = res[k].sample_details if hasattr(res[k], "sample_details") else []
                sampled_details = _ensure_user_ids(task, sampled_details)
                inference_time = per_lora_times.get(k)
                entry = dict(
                    results=res[k].aggregate_metrics,
                    sampled_res_details=sampled_details,
                    num_samples=len(sampled_details),
                    path=k,
                    inference_time_seconds=inference_time,
                    training_time_seconds=training_time_seconds,
                )
                results_payload[task].append(entry)
            results_payload.setdefault("_metadata", {})["timing"] = {
                "training_time_seconds": training_time_seconds,
                "total_inference_seconds": total_inference_seconds,
                "per_lora_inference_seconds": per_lora_times,
            }
            save_json(results_payload, result_path)
        torch.cuda.empty_cache()
        gc.collect()
