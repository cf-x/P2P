import os
import re
import datasets
import logging
import time
from functools import partial
from typing import Any, Callable, Literal, Optional, Sequence
import shutil
import json
from pathlib import Path

import vllm
import torch
from vllm.lora.request import LoRARequest
from transformers import set_seed
from transformers import AutoTokenizer

import fishfarm
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.models import GenerationRequest, Message
from fishfarm.tasks.language_restricted_math import (
    LanguageRestrictedMathTask,
    MathSample,
    extract_answer_number,
)
from fishfarm.tasks.evalplus import EvalplusTask, load_dataset
from fishfarm.tasks.rouge import RougeSample, RougeScorerConfig, RougeTask

from hyper_llm_modulator.utils.eval_tasks import (
    TASK_EVAL_FNS, 
    QASample, 
    QATask, 
    LaMPRatingTask, 
    LaMPClassificationTask, 
    RatingPredictionSample, 
    ClassificationSample, 
    TextGenerationSample, 
    LaMPTextGenerationTask,
    get_choice_accuracy,
    get_accuracy,
)
from hyper_llm_modulator.utils.eval_prompts import (
    IN_CONTEXT_EXAMPLES,
    TASK_DESCRIPTION_MESSAGES,
)
from hyper_llm_modulator.utils import get_preprocessing_fn, get_metadata_for_task

logger = logging.getLogger()

# Add OpinionQA to the task evaluation functions
TASK_EVAL_FNS["opinionqa"] = get_choice_accuracy


def copy_tokenizer_files_to_lora_dir(base_model_path: str, lora_dir: str):
    """Copy necessary tokenizer files from base model to LoRA directory."""
    try:
        # Load tokenizer from base model to get the files
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Save tokenizer to the LoRA directory
        tokenizer.save_pretrained(lora_dir)
        
        print(f"Copied tokenizer files to {lora_dir}")
        return True
    except Exception as e:
        print(f"Warning: Could not copy tokenizer files to {lora_dir}: {e}")
        return False


def _format_conversation_prompt(
    sample: dict,
    metadata: Optional[dict] = None,
    in_context_message: str = "",
) -> str:
    """Construct a textual prompt from conversational samples.

    The formatting aims to mimic the SFT formatting used during training by
    concatenating optional instructions, user profile text, and the conversation
    turns. Each turn is labeled with its speaker to maintain clarity.
    """

    metadata = metadata or {}
    sections: list[str] = []

    if in_context_message:
        sections.append(in_context_message.strip())

    task_description = metadata.get("task_description")
    if isinstance(task_description, str) and task_description.strip():
        sections.append(task_description.strip())

    if metadata.get("prepand_profile", False):
        profile_text = sample.get("profile_text") or sample.get("profile_all_history")
        if isinstance(profile_text, str) and profile_text.strip():
            sections.append(f"User Profile:\n{profile_text.strip()}")

    conversation = sample.get("conversation") or sample.get("input")
    if isinstance(conversation, list):
        convo_lines: list[str] = []
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            role = str(turn.get("role", "user")).lower()
            if "assistant" in role or "model" in role:
                speaker = "Assistant"
            elif "system" in role:
                speaker = "System"
            else:
                speaker = "User"
            convo_lines.append(f"{speaker}: {content}")
        if convo_lines:
            sections.append("\n".join(convo_lines))
    elif isinstance(conversation, str) and conversation.strip():
        sections.append(conversation.strip())

    prompt = "\n\n".join(section for section in sections if section)
    return prompt.strip()


def _infer_max_lora_rank(lora_dirs: Optional[Sequence[str]], default: int = 64) -> int:
    """Best-effort detection of the maximum LoRA rank present in the provided directories."""
    if not lora_dirs:
        return default

    detected_ranks: list[int] = []
    for lora_dir in lora_dirs:
        adapter_config = os.path.join(lora_dir, "adapter_config.json")
        if not os.path.exists(adapter_config):
            continue
        try:
            with open(adapter_config, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            rank = cfg.get("r") or cfg.get("lora_r") or cfg.get("rank")
            if rank is not None:
                detected_ranks.append(int(rank))
        except Exception as exc:
            logger.warning(f"Failed to parse LoRA rank from {adapter_config}: {exc}")

    return max(detected_ranks) if detected_ranks else default


class LoRAVLLMModel(VLLMModel):
    def __init__(self, prefill_text: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Store the original generate method to ensure clean LoRA switching
        self._original_generate = self.llm.generate

        if prefill_text is not None:
            # this can be used with chat_template
            f = self._into_prompt
            self._into_prompt = lambda messages: f(messages) + prefill_text

    def use_lora(self, lora_request: LoRARequest):
        # Reset to original generate method first, then apply new LoRA
        # This ensures we don't stack LoRA requests
        self.llm.generate = partial(self._original_generate, lora_request=lora_request)
        
    def clear_lora(self):
        """Reset to base model without any LoRA"""
        self.llm.generate = self._original_generate


@torch.no_grad()
def eval_model(
    model_dir,
    lora_dirs,
    chat_template,
    gpu_memory_utilization,
    evaluator,
    prefill_text="",
    per_sample_lora=True,
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    original_into_prompt = None
    wall_clock_start = time.perf_counter()
    timing_info: dict[str, Any] = {
        "per_lora_seconds": {},
        "per_call_seconds": [],
    }
    timing_info["started_at"] = time.time()

    def _record_time(lora_key: Optional[str], seconds: float) -> None:
        if lora_key is None:
            return
        aggregate = timing_info["per_lora_seconds"].get(lora_key, 0.0) + float(seconds)
        timing_info["per_lora_seconds"][lora_key] = aggregate
        timing_info["per_call_seconds"].append({"lora_dir": lora_key, "seconds": float(seconds)})

    if pre_created_model is not None:
        # Use the pre-created model instance
        model = pre_created_model
        if prefill_text != "":
            original_into_prompt = model._into_prompt
            model._into_prompt = lambda messages: original_into_prompt(messages) + prefill_text
    else:
        max_model_len = max_context_length or 2**12
        max_lora_rank = _infer_max_lora_rank(lora_dirs)

        llm_kwargs = dict(
            model=model_dir,
            seed=42,
            max_model_len=max_model_len,
            enable_lora=lora_dirs is not None,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if rope_scaling is not None:
            llm_kwargs["rope_scaling"] = rope_scaling

        kwargs = dict(
            llm=vllm.LLM(**llm_kwargs),
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=2**9,
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
            prefill_text=prefill_text,
        )
        model = LoRAVLLMModel(**kwargs)
    
    results = dict()
    if lora_dirs is not None:
        if per_sample_lora:
            assert len(lora_dirs) == evaluator.num_samples, (
                "Number of lora dirs must match number of samples when per_sample_lora is True"
            )
            # For per-sample LoRA, we need to evaluate each sample with its corresponding LoRA
            print(f"Evaluating {len(lora_dirs)} samples with their corresponding LoRAs")
            print(f"Total evaluator samples: {evaluator.num_samples}")
            
            # Create mapping from unique LoRA directories to LoRA requests
            # This handles user_id-based LoRAs where multiple samples share the same LoRA
            unique_lora_dirs = list(set(lora_dirs))
            
            # Copy tokenizer files to each unique LoRA directory if they don't exist
            print(f"Ensuring tokenizer files exist in LoRA directories...")
            for lora_dir in unique_lora_dirs:
                # Check if tokenizer files exist
                tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json']
                missing_files = [f for f in tokenizer_files if not os.path.exists(os.path.join(lora_dir, f))]
                
                if missing_files:
                    print(f"Missing tokenizer files in {lora_dir}: {missing_files}")
                    copy_tokenizer_files_to_lora_dir(model_dir, lora_dir)
            
            lora_dir_to_request = {}
            
            for idx, unique_lora_dir in enumerate(unique_lora_dirs):
                # Extract user_id from LoRA directory path for better naming
                if "user_" in unique_lora_dir:
                    # Extract user_id from path like ".../user_12345_lora_0"
                    dirname = os.path.basename(unique_lora_dir)
                    if dirname.startswith("user_"):
                        user_id_part = dirname.split("_lora_")[0]  # "user_12345"
                        lora_name = f"lora_{user_id_part}"
                    else:
                        lora_name = f"lora_unique_{idx}"
                else:
                    lora_name = f"lora_unique_{idx}"
                
                print("unique_lora_dir", unique_lora_dir)
                
                # Create LoRA request with user-based name
                lora_request = LoRARequest(lora_name, idx + 1, unique_lora_dir)
                lora_dir_to_request[unique_lora_dir] = lora_request
                print(f"Created LoRA request '{lora_name}' (ID: {idx + 1}) for {unique_lora_dir}")
            
            print(f"Created {len(unique_lora_dirs)} unique LoRA requests for {len(lora_dirs)} samples")
            
            # Map each LoRA directory to the sample indices that use it so we can batch per user.
            lora_dir_to_indices: dict[str, list[int]] = {}
            for sample_idx, lora_dir in enumerate(lora_dirs):
                lora_dir_to_indices.setdefault(lora_dir, []).append(sample_idx)

            # Allow tuning of the GPU batch size per LoRA via env var; default to a reasonable 8.
            per_lora_batch_env = os.getenv("P2P_VLLM_PER_LORA_BATCH", "256")
            try:
                per_lora_batch = max(1, int(per_lora_batch_env))
            except ValueError:
                per_lora_batch = 8
            print(f"Using per-LoRA batch size {per_lora_batch}")

            # We prefer to post-process using the task-specific batch evaluator when available.
            supports_batch_postproc = hasattr(evaluator, "batch_evaluate_with_outputs")
            outputs_by_index: list[str | None] = [None] * len(lora_dirs) if supports_batch_postproc else []
            sample_details_by_index: list[dict | None] = [None] * len(lora_dirs)

            chunk_results: list[tuple[list[int], Any]] = []

            def _chunk_indices(indices: list[int]) -> list[list[int]]:
                return [indices[i : i + per_lora_batch] for i in range(0, len(indices), per_lora_batch)]

            for unique_lora_dir, sample_indices in lora_dir_to_indices.items():
                print(
                    f"Evaluating {len(sample_indices)} samples for LoRA {unique_lora_dir}"
                )
                lora_request = lora_dir_to_request[unique_lora_dir]
                model.use_lora(lora_request)

                for chunk in _chunk_indices(sample_indices):
                    _eval_start = time.perf_counter()
                    chunk_result = evaluator.evaluate(model, chunk)
                    _record_time(unique_lora_dir, time.perf_counter() - _eval_start)
                    chunk_results.append((chunk, chunk_result))

                    for chunk_idx, detail in zip(chunk, chunk_result.sample_details):
                        sample_details_by_index[chunk_idx] = detail
                        if supports_batch_postproc:
                            outputs_by_index[chunk_idx] = detail.get("output")

            if supports_batch_postproc:
                if any(output is None for output in outputs_by_index):
                    missing = [i for i, output in enumerate(outputs_by_index) if output is None]
                    raise ValueError(
                        f"Missing generated outputs for sample indices {missing}; cannot batch evaluate"
                    )

                batch_outputs = [str(output) for output in outputs_by_index]
                print(
                    f"Performing batch evaluation on {len(batch_outputs)} samples across {len(lora_dir_to_indices)} LoRAs"
                )
                batch_result = evaluator.batch_evaluate_with_outputs(batch_outputs)
                results["per_sample_lora_combined"] = batch_result
                print(f"Batch evaluation results: {batch_result.aggregate_metrics}")
                print(f"Total samples evaluated: {len(batch_result.sample_details)}")
            else:
                # Fall back to aggregating chunk-level metrics when evaluators don't expose a batch API.
                print(
                    "Warning: Evaluator does not support batch evaluation, aggregating chunk metrics instead"
                )
                from fishfarm.tasks.base import TaskResult

                all_sample_details = [detail for detail in sample_details_by_index if detail is not None]
                metric_totals: dict[str, float] = {}
                metric_counts: dict[str, int] = {}

                for chunk, chunk_result in chunk_results:
                    for metric_name, metric_value in chunk_result.aggregate_metrics.items():
                        metric_totals.setdefault(metric_name, 0.0)
                        metric_counts.setdefault(metric_name, 0)
                        metric_totals[metric_name] += metric_value * len(chunk)
                        metric_counts[metric_name] += len(chunk)

                agg_metrics: dict[str, float] = {}
                if all_sample_details:
                    first_detail = all_sample_details[0]
                    if isinstance(first_detail, dict) and "is_correct" in first_detail:
                        total_correct = sum(detail.get("is_correct", 0) for detail in all_sample_details)
                        agg_metrics["acc"] = total_correct / len(all_sample_details)

                    for metric_name, total in metric_totals.items():
                        if metric_name == "acc" and metric_name in agg_metrics:
                            continue
                        count = metric_counts.get(metric_name, 0)
                        if count:
                            agg_metrics[metric_name] = total / count

                combined_result = TaskResult(
                    aggregate_metrics=agg_metrics,
                    sample_details=all_sample_details,
                )
                results["per_sample_lora_combined"] = combined_result
                print(f"Combined per-sample LoRA results: {agg_metrics}")
                print(f"Total samples evaluated: {len(all_sample_details)}")
        else:
            for i, lora_dir in enumerate(lora_dirs):
                print(f"Evaluating lora at: {lora_dir}")
                # Ensure tokenizer assets exist so vLLM does not fail when
                # attempting to load a tokenizer from the LoRA directory.
                tokenizer_marker = os.path.join(lora_dir, "tokenizer.json")
                if not os.path.exists(tokenizer_marker):
                    copy_tokenizer_files_to_lora_dir(model_dir, lora_dir)
                # NOTE: the second argument cannot be 0 or it will be treated as None
                # so painful trying to figure this out :(
                # also has to be unique
                lora_request = LoRARequest(f"lora_{i}", i + 1, lora_dir)
                model.use_lora(lora_request)
                _eval_start = time.perf_counter()
                results[lora_dir] = evaluator.evaluate(model)
                _record_time(lora_dir, time.perf_counter() - _eval_start)
    else:
        print(f"Evaluating base model at: {model_dir}")
        _eval_start = time.perf_counter()
        results[model_dir] = evaluator.evaluate(model)
        _record_time(model_dir, time.perf_counter() - _eval_start)
    
    # Clear any LoRA state and restore original _into_prompt if we modified it
    if hasattr(model, 'clear_lora'):
        model.clear_lora()
    if pre_created_model is not None and original_into_prompt is not None:
        model._into_prompt = original_into_prompt

    timing_info["total_inference_seconds"] = sum(item["seconds"] for item in timing_info["per_call_seconds"])
    timing_info["wall_clock_seconds"] = time.perf_counter() - wall_clock_start
    timing_info["num_calls"] = len(timing_info["per_call_seconds"])
    timing_info["completed_at"] = time.time()

    return {"results": results, "timing": timing_info}


@torch.no_grad()
def eval_task_from_config(
    model_dir: str,
    task_name: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    pre_created_model=None,
    in_context_message: str = "",
    retrieval_k: Optional[int] = None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    """
    Evaluate a task using configuration from the tasks directory.
    
    Args:
        model_dir: Path to the base model
        task_name: Name of the task (should match directory name in tasks/)
        lora_dirs: List of LoRA directories to evaluate
        chat_template: Chat template to use
        gpu_memory_utilization: GPU memory utilization
        prefill_text: Text to prefill in generation
        per_sample_lora: Whether to use per-sample LoRA
        retrieval_k: Number of retrieved history entries to include when templates support it
        max_context_length: Optional override for vLLM max context length
        rope_scaling: Optional rope scaling configuration (e.g., YaRN)
    """
    # Load task metadata
    metadata = get_metadata_for_task(task_name)
    
    # Load dataset
    dataset_path = metadata["ds_kwargs"]["path"]
    dataset_name = metadata["ds_kwargs"]["name"]
    
    # Load the dataset from the path
    dataset = datasets.load_from_disk(dataset_path)
    if isinstance(dataset, datasets.DatasetDict):
        split = metadata["ds_kwargs"].get("split") if isinstance(metadata.get("ds_kwargs"), dict) else None
        if split and split in dataset:
            dataset = dataset[split]
        elif dataset_name and dataset_name in dataset:
            dataset = dataset[dataset_name]
        else:
            first_split = next(iter(dataset.keys()))
            dataset = dataset[first_split]
    
    # Get preprocessing function if needed
    preprocessing_fn = get_preprocessing_fn(dataset_name)
    if preprocessing_fn is not None:
        dataset = dataset.map(preprocessing_fn, batched=False)
    
    # Extract template information
    system_message = metadata.get("system_message", "")
    user_prompt_template = metadata.get("user_prompt_template")
    response_field = metadata.get("response_field", "output")
    assistant_prefill = metadata.get("assistant_prefill", "")

    # Allow dynamic selection of retrieval history fields when requested
    selected_retrieval_field = None
    if retrieval_k is not None and user_prompt_template:
        candidate_template = user_prompt_template
        selected_retrieval_field = f"profile_retrieval_k{retrieval_k}"
        if isinstance(user_prompt_template, str):
            candidate_template = re.sub(
                r"profile_retrieval_k\d+", selected_retrieval_field, user_prompt_template
            )

        if selected_retrieval_field not in dataset.column_names:
            if "profile_text" in dataset.column_names:
                logger.warning(
                    f"Requested retrieval field '{selected_retrieval_field}' missing in dataset for task {task_name}. "
                    "Using 'profile_text' fallback and switching to gen_profile format."
                )

                def _inject_profile_text(sample):
                    sample[selected_retrieval_field] = str(sample.get("profile_text", ""))
                    sample["user_profile_format"] = "gen_profile"
                    return sample

                dataset = dataset.map(
                    _inject_profile_text,
                    load_from_cache_file=False,
                    desc=f"Fallback to profile_text for {selected_retrieval_field}",
                )
                metadata["user_profile_format"] = "gen_profile"
            else:
                logger.warning(
                    f"Requested retrieval field '{selected_retrieval_field}' missing and no 'profile_text' fallback available for task {task_name}."
                )
                selected_retrieval_field = None

        if isinstance(user_prompt_template, str):
            user_prompt_template = candidate_template

    
    # Determine the evaluation type based on task name
    if user_prompt_template is None:
        return eval_conversation_task(
            model_dir=model_dir,
            lora_dirs=lora_dirs,
            chat_template=chat_template,
            gpu_memory_utilization=gpu_memory_utilization,
            metadata=metadata,
            dataset=dataset,
            response_field=response_field,
            prefill_text=prefill_text,
            per_sample_lora=per_sample_lora,
            pre_created_model=pre_created_model,
            in_context_message=in_context_message,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
        )

    task_lower = task_name.lower()
    dataset_name_lower = (dataset_name or "").lower()
    dataset_path_parts = [part.lower() for part in Path(str(dataset_path)).parts if part]
    is_personalreddit = (
        "personalreddit" in task_lower
        or "personalreddit" in dataset_name_lower
        or any("personalreddit" in part for part in dataset_path_parts)
    )
    is_ec = (
        task_lower.startswith("ec")
        or dataset_name_lower.startswith("ec")
        or any(part == "ec" for part in dataset_path_parts)
    )

    if is_personalreddit or is_ec:
        return eval_text_generation_task(
            model_dir=model_dir,
            lora_dirs=lora_dirs,
            chat_template=chat_template,
            gpu_memory_utilization=gpu_memory_utilization,
            system_message=system_message,
            template=user_prompt_template,
            dataset=dataset,
            response_field=response_field,
            prefill_text=prefill_text or assistant_prefill,
            per_sample_lora=per_sample_lora,
            pre_created_model=pre_created_model,
            in_context_message=in_context_message,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
        )

    if "opinionqa" in task_name.lower():
        return eval_classification_task(
            model_dir=model_dir,
            lora_dirs=lora_dirs,
            chat_template=chat_template,
            gpu_memory_utilization=gpu_memory_utilization,
            system_message=system_message,
            template=user_prompt_template,
            dataset=dataset,
            response_field=response_field,
            prefill_text=prefill_text or assistant_prefill,
            per_sample_lora=per_sample_lora,
            eval_fn=TASK_EVAL_FNS["opinionqa"],
            pre_created_model=pre_created_model,
            in_context_message=in_context_message,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
        )

    elif any(x in task_name.lower() for x in ["lamp_scholarly_title", "lamp_tweet", "lamp_news_headline", "longlamp_abstract_generation", "longlamp_product_review", "longlamp_topic_writing"]):
        # Text generation tasks
        return eval_text_generation_task(
            model_dir=model_dir,
            lora_dirs=lora_dirs,
            chat_template=chat_template,
            gpu_memory_utilization=gpu_memory_utilization,
            system_message=system_message,
            template=user_prompt_template,
            dataset=dataset,
            response_field=response_field,
            prefill_text=prefill_text or assistant_prefill,
            per_sample_lora=per_sample_lora,
            pre_created_model=pre_created_model,
            in_context_message=in_context_message,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
        )

    elif any(x in task_name.lower() for x in ["lamp_product", "lamp_3"]):
        # Rating prediction tasks
        return eval_rating_task(
            model_dir=model_dir,
            lora_dirs=lora_dirs,
            chat_template=chat_template,
            gpu_memory_utilization=gpu_memory_utilization,
            system_message=system_message,
            template=user_prompt_template,
            dataset=dataset,
            response_field=response_field,
            prefill_text=prefill_text or assistant_prefill,
            per_sample_lora=per_sample_lora,
            pre_created_model=pre_created_model,
            in_context_message=in_context_message,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
        )
    elif any(x in task_name.lower() for x in ["lamp_movie", "lamp_citation", "lamp_news_cat"]):
        # Classification tasks
        return eval_classification_task(
            model_dir=model_dir,
            lora_dirs=lora_dirs,
            chat_template=chat_template,
            gpu_memory_utilization=gpu_memory_utilization,
            system_message=system_message,
            template=user_prompt_template,
            dataset=dataset,
            response_field=response_field,
            prefill_text=prefill_text or assistant_prefill,
            per_sample_lora=per_sample_lora,
            eval_fn=get_accuracy,
            pre_created_model=pre_created_model,
            in_context_message=in_context_message,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
        )
    
    else:
        # Default to Q&A evaluation
        return eval_qa_task_from_config(
            model_dir=model_dir,
            lora_dirs=lora_dirs,
            chat_template=chat_template,
            gpu_memory_utilization=gpu_memory_utilization,
            system_message=system_message,
            template=user_prompt_template,
            dataset=dataset,
            response_field=response_field,
            prefill_text=prefill_text or assistant_prefill,
            per_sample_lora=per_sample_lora,
            eval_fn=get_accuracy,
            pre_created_model=pre_created_model,
            in_context_message=in_context_message,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
        )


@torch.no_grad()
def eval_qa_task_from_config(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    dataset=None,
    response_field: str = "output",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    eval_fn: Callable = get_accuracy,
    pre_created_model=None,
    in_context_message: str = "",
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    """Evaluate Q&A tasks using configuration from metadata"""
    samples = []
    for sample in dataset:
        # Format the prompt using the template and preprocessed sample
        question = template.format(**sample)
        if in_context_message:
            question = in_context_message + "\n\n" + question
        answer = str(sample[response_field])
        samples.append(QASample(question=question, answer=answer))

    evaluator = QATask(
        samples=samples,
        eval_fn=eval_fn,
        context_messages=[fishfarm.Message("system", system_message)] if system_message else [],
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_classification_task(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    dataset=None,
    response_field: str = "output",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    eval_fn: Callable = get_accuracy,
    pre_created_model=None,
    in_context_message: str = "",
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    """Evaluate classification tasks"""
    samples = []
    for sample in dataset:
        prompt = template.format(**sample)
        if in_context_message:
            prompt = in_context_message + "\n\n" + prompt
        label = str(sample[response_field])
        user_profile = sample.get("profile_text") or sample.get("profile_all_history")
        samples.append(
            ClassificationSample(
                prompt=prompt,
                label=label,
                user_profile=user_profile,
                user_id=sample.get("user_id"),
                question_id=sample.get("question_id"),
            )
        )

    evaluator = LaMPClassificationTask(
        samples=samples,
        context_messages=[fishfarm.Message("system", system_message)] if system_message else [],
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_rating_task(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    dataset=None,
    response_field: str = "output",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    pre_created_model=None,
    in_context_message: str = "",
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    """Evaluate rating prediction tasks"""
    samples = []
    for sample in dataset:
        prompt = template.format(**sample)
        if in_context_message:
            prompt = in_context_message + "\n\n" + prompt
        rating_value = sample[response_field]
        if isinstance(rating_value, str):
            try:
                rating_value = float(rating_value)
            except ValueError:
                continue  # Skip samples with invalid ratings
        user_profile = sample.get("profile_text") or sample.get("profile_all_history")
        samples.append(
            RatingPredictionSample(
                prompt=prompt,
                rating=float(rating_value),
                user_profile=user_profile,
                user_id=sample.get("user_id"),
                question_id=sample.get("question_id"),
            )
        )

    evaluator = LaMPRatingTask(
        samples=samples,
        context_messages=[fishfarm.Message("system", system_message)] if system_message else [],
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_text_generation_task(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    dataset=None,
    response_field: str = "output",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    rouge_types: tuple = ("rouge1", "rougeL"),
    pre_created_model=None,
    in_context_message: str = "",
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    """Evaluate text generation tasks"""
    samples = []
    for sample in dataset:
        prompt = template.format(**sample)
        if in_context_message:
            prompt = in_context_message + "\n\n" + prompt
        reference = str(sample[response_field])
        user_profile = sample.get("profile_text") or sample.get("profile_all_history")
        samples.append(
            TextGenerationSample(
                prompt=prompt,
                reference=reference,
                user_profile=user_profile,
                user_id=sample.get("user_id"),
                question_id=sample.get("question_id"),
            )
        )

    evaluator = LaMPTextGenerationTask(
        samples=samples,
        context_messages=[fishfarm.Message("system", system_message)] if system_message else [],
        rouge_types=rouge_types,
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_conversation_task(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    metadata: Optional[dict] = None,
    dataset=None,
    response_field: str = "output",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    pre_created_model=None,
    in_context_message: str = "",
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
) -> dict:
    """Evaluate personalization chat-style tasks without explicit templates."""

    metadata = metadata or {}
    system_message = metadata.get("system_message", "")
    assistant_prefill = prefill_text if prefill_text is not None else metadata.get("assistant_prefill", "")

    samples: list[TextGenerationSample] = []
    for sample in dataset:
        prompt = _format_conversation_prompt(sample, metadata, in_context_message)
        reference = str(sample.get(response_field, ""))
        text_sample = TextGenerationSample(prompt=prompt, reference=reference)

        # Attach auxiliary metadata for downstream analysis when available
        if sample.get("profile_text"):
            text_sample.user_profile = sample["profile_text"]
        if sample.get("user_id"):
            text_sample.user_id = sample["user_id"]
        if sample.get("question_id"):
            text_sample.question_id = sample["question_id"]

        samples.append(text_sample)

    evaluator = LaMPTextGenerationTask(
        samples=samples,
        context_messages=[fishfarm.Message("system", system_message)] if system_message else [],
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        assistant_prefill,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_rouge(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    in_context_message: str = "",
    dataset_name: str | None = None,
    dataset_kwargs: dict = {},
    preprocessing_fn: Optional[callable] = None,
    response_field: str = "",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    dataset = datasets.load_dataset(
        dataset_name, **dataset_kwargs
    )  # , download_mode="force_redownload")
    if preprocessing_fn is not None:
        dataset = dataset.map(preprocessing_fn, batched=False)
    samples = []
    for sample in dataset:
        prompt = template.format(**sample)
        if in_context_message:
            prompt = in_context_message + "\n\n" + prompt
        samples.append(RougeSample(prompt=prompt, response=str(sample[response_field])))
    evaluator = RougeTask(
        samples=samples,
        context_messages=[fishfarm.Message("system", system_message)],
        rouge_scorer_config=RougeScorerConfig(use_stemmer=True),
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_qa(
    model_dir: str,
    eval_fn: Callable,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    in_context_message: str = "",
    dataset_name: str | None = None,
    dataset_kwargs: dict = {},
    preprocessing_fn: Optional[callable] = None,
    response_field: str = "",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    dataset = datasets.load_dataset(dataset_name, **dataset_kwargs)
    if preprocessing_fn is not None:
        dataset = dataset.map(preprocessing_fn, batched=False)
    samples = []
    for sample in dataset:
        question = template.format(**sample)
        if in_context_message:
            question = in_context_message + "\n\n" + question
        answer = str(sample[response_field])
        samples.append(QASample(question=question, answer=answer))

    evaluator = QATask(
        samples=samples,
        eval_fn=eval_fn,
        context_messages=[fishfarm.Message("system", system_message)],
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_lamp_rating(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    in_context_message: str = "",
    dataset_name: str | None = None,
    dataset_kwargs: dict = {},
    preprocessing_fn: Optional[callable] = None,
    rating_field: str = "rating",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    """Evaluate LaMP rating prediction tasks with MAE and RMSE metrics."""
    dataset = datasets.load_dataset(dataset_name, **dataset_kwargs)
    if preprocessing_fn is not None:
        dataset = dataset.map(preprocessing_fn, batched=False)
    
    samples = []
    for sample in dataset:
        prompt = template.format(**sample)
        if in_context_message:
            prompt = in_context_message + "\n\n" + prompt
        # Convert rating to float, handle various formats
        rating_value = sample[rating_field]
        if isinstance(rating_value, str):
            try:
                rating_value = float(rating_value)
            except ValueError:
                continue  # Skip samples with invalid ratings
        user_profile = sample.get("profile_text") or sample.get("profile_all_history")
        samples.append(
            RatingPredictionSample(
                prompt=prompt,
                rating=float(rating_value),
                user_profile=user_profile,
                user_id=sample.get("user_id"),
                question_id=sample.get("question_id"),
            )
        )
    
    evaluator = LaMPRatingTask(
        samples=samples,
        context_messages=[fishfarm.Message("system", system_message)],
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_lamp_classification(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    in_context_message: str = "",
    dataset_name: str | None = None,
    dataset_kwargs: dict = {},
    preprocessing_fn: Optional[callable] = None,
    label_field: str = "label",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    """Evaluate LaMP classification tasks with accuracy and F1 metrics."""
    dataset = datasets.load_dataset(dataset_name, **dataset_kwargs)
    if preprocessing_fn is not None:
        dataset = dataset.map(preprocessing_fn, batched=False)
    
    samples = []
    for sample in dataset:
        prompt = template.format(**sample)
        if in_context_message:
            prompt = in_context_message + "\n\n" + prompt
        user_profile = sample.get("profile_text") or sample.get("profile_all_history")
        samples.append(
            ClassificationSample(
                prompt=prompt,
                label=str(sample[label_field]),
                user_profile=user_profile,
                user_id=sample.get("user_id"),
                question_id=sample.get("question_id"),
            )
        )
    
    evaluator = LaMPClassificationTask(
        samples=samples,
        context_messages=[fishfarm.Message("system", system_message)],
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_lamp_text_generation(
    model_dir: str,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    gpu_memory_utilization: float = 0.7,
    system_message: str = "",
    template: str = "",
    in_context_message: str = "",
    dataset_name: str | None = None,
    dataset_kwargs: dict = {},
    preprocessing_fn: Optional[callable] = None,
    response_field: str = "",
    prefill_text: Optional[str] = None,
    per_sample_lora: bool = False,
    rouge_types: tuple = ("rouge1", "rougeL"),
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    """Evaluate LaMP text generation tasks with ROUGE-1, ROUGE-L, and METEOR metrics."""
    dataset = datasets.load_dataset(dataset_name, **dataset_kwargs)
    if preprocessing_fn is not None:
        dataset = dataset.map(preprocessing_fn, batched=False)
    
    samples = []
    for sample in dataset:
        prompt = template.format(**sample)
        if in_context_message:
            prompt = in_context_message + "\n\n" + prompt
        reference = str(sample[response_field])
        user_profile = sample.get("profile_text") or sample.get("profile_all_history")
        samples.append(
            TextGenerationSample(
                prompt=prompt,
                reference=reference,
                user_profile=user_profile,
                user_id=sample.get("user_id"),
                question_id=sample.get("question_id"),
            )
        )
    
    evaluator = LaMPTextGenerationTask(
        samples=samples,
        context_messages=[fishfarm.Message("system", system_message)],
        rouge_types=rouge_types,
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text,
        per_sample_lora,
        pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_gsm8k(
    model_dir,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    in_context_message: str = "",
    gpu_memory_utilization=0.7,
    per_sample_lora: bool = False,
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    system_message = ""
    template = "Please answer the following question: {question}\n\n"
    dataset = datasets.load_dataset("gsm8k", "main", split="test")
    samples = []
    for sample in dataset:
        problem = template.format(**sample)
        if in_context_message:
            problem = in_context_message + "\n\n" + problem
        answer = sample["answer"]
        answer = extract_answer_number(answer)
        answer = int(answer) if answer is not None else None
        samples.append(
            MathSample(
                problem=template.format(**sample),
                answer=answer,
            )
        )

    evaluator = LanguageRestrictedMathTask(
        samples=samples,
        context_messages=[
            fishfarm.Message("system", system_message),
        ],
        languages=[],  # No need for language detection since the GSM8K task is purely in English.
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text="Let's think step by step.",
        per_sample_lora=per_sample_lora,
        pre_created_model=pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


@torch.no_grad()
def eval_coding(
    model_dir,
    lora_dirs: Optional[list[str]] = None,
    chat_template: Optional[str] = None,
    in_context_message: str = "",
    gpu_memory_utilization=0.7,
    source_dataset: Literal["humaneval", "mbpp"] = "humaneval",
    per_sample_lora: bool = False,
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
):
    system_message = ""
    samples = load_dataset(source_dataset=source_dataset)
    if in_context_message:
        for sample in samples:
            sample.instruction = in_context_message + "\n\n" + sample.instruction

    evaluator = EvalplusTask(
        samples,
        context_messages=[fishfarm.Message("system", system_message)],
        source_dataset=source_dataset,
    )

    return eval_model(
        model_dir,
        lora_dirs,
        chat_template,
        gpu_memory_utilization,
        evaluator,
        prefill_text="",
        per_sample_lora=per_sample_lora,
        pre_created_model=pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )


OQA_TEMPLATE = (
    "Complete the following passage or answer the question by choosing the correct choice.\n\n"
    "{question_stem}\n\n"
    "{choices[label][0]}: {choices[text][0]}\n{choices[label][1]}: {choices[text][1]}\n"
    "{choices[label][2]}: {choices[text][2]}\n{choices[label][3]}: {choices[text][3]}\n\n"
    "You must respond with the letter corresponding to the correct choice (A,B,C,D) without any explanation."
)
ARC_TEMPLATE = (
    "Answer the question below by choosing the correct choice.\n\n"
    "{question}\n\n"
    "{choices[label][0]}: {choices[text][0]}\n{choices[label][1]}: {choices[text][1]}\n"
    "{choices[label][2]}: {choices[text][2]}\n{choices[label][3]}: {choices[text][3]}\n\n"
    "You must respond with the letter corresponding to the correct choice without any explanation."
)
HSWAG_TEMPLATE = (
    "You are provided with an incomplete passage below as well as 4 choices of continuation "
    "with only one of them being the correct ending. "
    "Treat the endings as being labelled 0, 1, 2, 3 in order.\n\n"
    "Passage: {ctx}\n\n"
    "0: {endings[0]}\n"
    "1: {endings[1]}\n"
    "2: {endings[2]}\n"
    "3: {endings[3]}\n\n"
    "You must respond with the only number corresponding to the correct ending (0,1,2,3) for the passage "
    "without any explanation."
)
PIQA_TEMPLATE = (
    "Choose the option that either answers the question, completes the sentence, or solves the problem. "
    "Pay attention to the properties of the objects in the question and how they interact with each other. "
    'If both options are correct, choose the one that is more convenient or more common.\n\n"""{goal}"""\n\n'
    "0: {sol1}\n1: {sol2}\n\n"
    "You must respond with either `0` or `1` without any explanation."
)
WINOGRANDE_TEMPLATE = (
    "Given the following situation:\n\n{sentence}\n\nWhich option is correct?\n\n"
    "Option 1: {option1}\n\nOption 2: {option2}\n\n"
    "You must respond with either `1` or `2` without any explanation."
)
FIQA_SYSTEM_MESSAGE = ""

FIQA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately answer user input.\n\n"
    "Instruction:{instruction}\n\nInput:{input}"
)

BOOLQ_SYSTEM_MESSAGE = ""
BOOLQ_TEMPLATE = "{passage}\n\nQuestion: {question}?\n\nPlease answer with either `true` or `false` without any explanation."
LOL_TEMPLATE = "{task_def}\n\n{problem}"

TASK_TEMPLATES = {
    "gsm8k": "Please answer the following question: {question}\n\n",
    "boolq": BOOLQ_TEMPLATE,
    "winogrande": WINOGRANDE_TEMPLATE,
    "piqa": PIQA_TEMPLATE,
    "hellaswag": HSWAG_TEMPLATE,
    "arc_easy": ARC_TEMPLATE,
    "arc_challenge": ARC_TEMPLATE,
    "openbookqa": OQA_TEMPLATE,
    "context_alphabet": "{query}",
    "context_numbers": "{query}",
    "mini_pwc": "{query}",
}

DS_PATHS = {
    "gsm8k": "openai/gsm8k",
    "boolq": "google/boolq",
    "winogrande": "allenai/winogrande",
    "piqa": "ybisk/piqa",
    "hellaswag": "Rowan/hellaswag",
    "arc_easy": "allenai/ai2_arc",
    "arc_challenge": "allenai/ai2_arc",
    "openbookqa": "allenai/openbookqa",
    "context_alphabet": "data/raw_datasets/context_alphabet",
    "context_numbers": "data/raw_datasets/context_numbers",
    "mini_pwc": "sggetao/PwC",
}

DS_KWARGS = {
    "gsm8k": dict(name="main", split="test"),
    "boolq": dict(split="validation"),
    "winogrande": dict(name="winogrande_debiased", split="validation"),
    "piqa": dict(split="validation"),
    "hellaswag": dict(split="validation"),
    "arc_easy": dict(name="ARC-Easy", split="test"),
    "arc_challenge": dict(name="ARC-Challenge", split="test"),
    "openbookqa": dict(split="test"),
}

EVAL_FNS = {
    "gsm8k": eval_gsm8k,
    "humaneval": partial(eval_coding, source_dataset="humaneval"),
    "mbpp": partial(eval_coding, source_dataset="mbpp"),
    "fiqa": partial(
        eval_rouge,
        system_message=FIQA_SYSTEM_MESSAGE,
        template=FIQA_TEMPLATE,
        dataset_name="llamafactory/fiqa",
        dataset_kwargs=dict(split="test"),
        response_field="output",
    ),
    "boolq": partial(
        eval_qa,
        system_message="",
        template=BOOLQ_TEMPLATE,
        dataset_name="google/boolq",
        dataset_kwargs=dict(split="validation"),
        response_field="answer",
        prefill_text="",
        eval_fn=TASK_EVAL_FNS["boolq"],
    ),
    "winogrande": partial(
        eval_qa,
        system_message="",
        template=WINOGRANDE_TEMPLATE,
        dataset_name="allenai/winogrande",
        dataset_kwargs=dict(
            name="winogrande_debiased", split="validation", trust_remote_code=True
        ),
        response_field="answer",
        prefill_text="",
        eval_fn=TASK_EVAL_FNS["winogrande"],
    ),
    "piqa": partial(
        eval_qa,
        system_message="",
        template=PIQA_TEMPLATE,
        dataset_name="ybisk/piqa",
        # test split does not have labels
        dataset_kwargs=dict(split="validation", trust_remote_code=True),
        response_field="label",
        prefill_text="",
        eval_fn=TASK_EVAL_FNS["piqa"],
    ),
    "hellaswag": partial(
        eval_qa,
        system_message="",
        template=HSWAG_TEMPLATE,
        dataset_name="Rowan/hellaswag",
        # test split does not have labels
        dataset_kwargs=dict(split="validation"),
        response_field="label",
        prefill_text="",
        eval_fn=TASK_EVAL_FNS["hellaswag"],
    ),
    "arc_easy": partial(
        eval_qa,
        system_message="",
        template=ARC_TEMPLATE,
        dataset_name="allenai/ai2_arc",
        dataset_kwargs=dict(name="ARC-Easy", split="test"),
        response_field="answerKey",
        prefill_text="",
        preprocessing_fn=get_preprocessing_fn("arc_easy"),
        eval_fn=TASK_EVAL_FNS["arc"],
    ),
    "arc_challenge": partial(
        eval_qa,
        system_message="",
        template=ARC_TEMPLATE,
        dataset_name="allenai/ai2_arc",
        dataset_kwargs=dict(name="ARC-Challenge", split="test"),
        response_field="answerKey",
        prefill_text="",
        preprocessing_fn=get_preprocessing_fn("arc_challenge"),
        eval_fn=TASK_EVAL_FNS["arc"],
    ),
    "openbookqa": partial(
        eval_qa,
        system_message="",
        template=OQA_TEMPLATE,
        dataset_name="allenai/openbookqa",
        dataset_kwargs=dict(split="test"),
        response_field="answerKey",
        prefill_text="",
        eval_fn=TASK_EVAL_FNS["openbookqa"],
    ),
    # Personalization evaluation tasks - using task config approach
    "opinionqa_random_test": partial(eval_task_from_config, task_name="opinionqa_random_test"),
    "opinionqa_random_train": partial(eval_task_from_config, task_name="opinionqa_random_train"),
    "lamp_movie_random_test": partial(eval_task_from_config, task_name="lamp_movie_random_test"),
    "lamp_movie_random_train": partial(eval_task_from_config, task_name="lamp_movie_random_train"),
    "lamp_citation_random_test": partial(eval_task_from_config, task_name="lamp_citation_random_test"),
    "lamp_citation_random_train": partial(eval_task_from_config, task_name="lamp_citation_random_train"),
    "lamp_product_random_test": partial(eval_task_from_config, task_name="lamp_product_random_test"),
    "lamp_product_random_train": partial(eval_task_from_config, task_name="lamp_product_random_train"),
    "lamp_tweet_random_test": partial(eval_task_from_config, task_name="lamp_tweet_random_test"),
    "lamp_tweet_random_train": partial(eval_task_from_config, task_name="lamp_tweet_random_train"),
    "lamp_news_headline_random_test": partial(eval_task_from_config, task_name="lamp_news_headline_random_test"),
    "lamp_news_headline_random_train": partial(eval_task_from_config, task_name="lamp_news_headline_random_train"),
    "lamp_news_cat_random_test": partial(eval_task_from_config, task_name="lamp_news_cat_random_test"),
    "lamp_news_cat_random_train": partial(eval_task_from_config, task_name="lamp_news_cat_random_train"),
    "lamp_scholarly_title_random_test": partial(eval_task_from_config, task_name="lamp_scholarly_title_random_test"),
    "lamp_scholarly_title_random_train": partial(eval_task_from_config, task_name="lamp_scholarly_title_random_train"),
    "longlamp_abstract_generation_random_test": partial(eval_task_from_config, task_name="longlamp_abstract_generation_random_test"),
    "longlamp_abstract_generation_random_train": partial(eval_task_from_config, task_name="longlamp_abstract_generation_random_train"),
    "longlamp_product_review_random_test": partial(eval_task_from_config, task_name="longlamp_product_review_random_test"),
    "longlamp_product_review_random_train": partial(eval_task_from_config, task_name="longlamp_product_review_random_train"),
    "longlamp_topic_writing_random_test": partial(eval_task_from_config, task_name="longlamp_topic_writing_random_test"),
    "longlamp_topic_writing_random_train": partial(eval_task_from_config, task_name="longlamp_topic_writing_random_train"),
    "personalreddit_random_test": partial(eval_task_from_config, task_name="personalreddit_random_test"),
    "personalreddit_random_train": partial(eval_task_from_config, task_name="personalreddit_random_train"),
    "prism_random_test": partial(eval_task_from_config, task_name="prism_random_test"),
    "prism_random_train": partial(eval_task_from_config, task_name="prism_random_train"),
    "EC_random_test": partial(eval_task_from_config, task_name="EC_random_test"),
    "EC_random_train": partial(eval_task_from_config, task_name="EC_random_train"),
    "aloe_random_test": partial(eval_task_from_config, task_name="aloe_random_test"),
    "aloe_random_train": partial(eval_task_from_config, task_name="aloe_random_train"),
    "aloe_ood_test": partial(eval_task_from_config, task_name="aloe_ood_test"),
    "aloe_ood_train": partial(eval_task_from_config, task_name="aloe_ood_train"),
    "opinionqa_train": partial(eval_task_from_config, task_name="opinionqa_train"),
    "lamp_movie_train": partial(eval_task_from_config, task_name="lamp_movie_train"),
    "lamp_citation_train": partial(eval_task_from_config, task_name="lamp_citation_train"),
    "lamp_product_train": partial(eval_task_from_config, task_name="lamp_product_train"),
    "lamp_tweet_train": partial(eval_task_from_config, task_name="lamp_tweet_train"),
    "lamp_news_headline_train": partial(eval_task_from_config, task_name="lamp_news_headline_train"),
    "lamp_news_cat_train": partial(eval_task_from_config, task_name="lamp_news_cat_train"),
    "lamp_scholarly_title_train": partial(eval_task_from_config, task_name="lamp_scholarly_title_train"),
    "longlamp_abstract_generation_train": partial(eval_task_from_config, task_name="longlamp_abstract_generation_train"),
    "longlamp_product_review_train": partial(eval_task_from_config, task_name="longlamp_product_review_train"),
    "longlamp_topic_writing_train": partial(eval_task_from_config, task_name="longlamp_topic_writing_train"),
    "longlamp_topic_writing_train_sub": partial(eval_task_from_config, task_name="longlamp_topic_writing_train_sub"),
    "personalreddit_train": partial(eval_task_from_config, task_name="personalreddit_train"),
    "prism_train": partial(eval_task_from_config, task_name="prism_train"),
    "EC_train": partial(eval_task_from_config, task_name="EC_train"),
    "aloe_train": partial(eval_task_from_config, task_name="aloe_train")
}


lol_paths = [
    "Lots-of-LoRAs/task742_lhoestq_answer_generation_frequency",
    "Lots-of-LoRAs/task1198_atomic_classification_owant",
    "Lots-of-LoRAs/task717_mmmlu_answer_generation_logical_fallacies",
    "Lots-of-LoRAs/task705_mmmlu_answer_generation_high_school_macroeconomics",
    "Lots-of-LoRAs/task275_enhanced_wsc_paraphrase_generation",
    "Lots-of-LoRAs/task636_extract_and_sort_unique_alphabets_in_a_list",
    "Lots-of-LoRAs/task084_babi_t1_single_supporting_fact_identify_relevant_fact",
    "Lots-of-LoRAs/task1711_poki_text_generation",
    "Lots-of-LoRAs/task140_detoxifying-lms_classification_style",
    "Lots-of-LoRAs/task1448_disease_entity_extraction_ncbi_dataset",
    "Lots-of-LoRAs/task453_swag_answer_generation",
    "Lots-of-LoRAs/task1207_atomic_classification_atlocation",
    "Lots-of-LoRAs/task734_mmmlu_answer_generation_sociology",
    "Lots-of-LoRAs/task298_storycloze_correct_end_classification",
    "Lots-of-LoRAs/task587_amazonfood_polarity_correction_classification",
    "Lots-of-LoRAs/task703_mmmlu_answer_generation_high_school_geography",
    "Lots-of-LoRAs/task147_afs_argument_similarity_gay_marriage",
    "Lots-of-LoRAs/task564_discofuse_classification",
    "Lots-of-LoRAs/task1341_msr_text_classification",
    "Lots-of-LoRAs/task201_mnli_neutral_classification",
    "Lots-of-LoRAs/task890_gcwd_classification",
    "Lots-of-LoRAs/task908_dialogre_identify_familial_relationships",
    "Lots-of-LoRAs/task1428_country_surface_area",
    "Lots-of-LoRAs/task202_mnli_contradiction_classification",
    "Lots-of-LoRAs/task325_jigsaw_classification_identity_attack",
    "Lots-of-LoRAs/task1669_md_gender_bias_text_modification",
    "Lots-of-LoRAs/task246_dream_question_generation",
    "Lots-of-LoRAs/task357_casino_classification_negotiation_small_talk",
    "Lots-of-LoRAs/task1518_limit_answer_generation",
    "Lots-of-LoRAs/task1148_maximum_ascii_value",
    "Lots-of-LoRAs/task1605_ethos_text_classification",
    "Lots-of-LoRAs/task867_mawps_multiop_question_answering",
    "Lots-of-LoRAs/task209_stancedetection_classification",
    "Lots-of-LoRAs/task751_svamp_subtraction_question_answering",
    "Lots-of-LoRAs/task161_count_words_containing_letter",
    "Lots-of-LoRAs/task105_story_cloze-rocstories_sentence_generation",
    "Lots-of-LoRAs/task645_summarization",
    "Lots-of-LoRAs/task442_com_qa_paraphrase_question_generation",
    "Lots-of-LoRAs/task075_squad1.1_answer_generation",
    "Lots-of-LoRAs/task269_csrg_counterfactual_story_generation",
    "Lots-of-LoRAs/task1568_propara_classification",
    "Lots-of-LoRAs/task834_mathdataset_classification",
    "Lots-of-LoRAs/task1603_smcalflow_sentence_generation",
    "Lots-of-LoRAs/task685_mmmlu_answer_generation_clinical_knowledge",
    "Lots-of-LoRAs/task083_babi_t1_single_supporting_fact_answer_generation",
    "Lots-of-LoRAs/task390_torque_text_span_selection",
    "Lots-of-LoRAs/task750_aqua_multiple_choice_answering",
    "Lots-of-LoRAs/task1631_openpi_answer_generation",
    "Lots-of-LoRAs/task1529_scitail1.1_classification",
    "Lots-of-LoRAs/task746_yelp_restaurant_review_classification",
    "Lots-of-LoRAs/task1217_atomic_answer_generation",
    "Lots-of-LoRAs/task725_mmmlu_answer_generation_nutrition",
    "Lots-of-LoRAs/task039_qasc_find_overlapping_words",
    "Lots-of-LoRAs/task889_goemotions_classification",
    "Lots-of-LoRAs/task492_mwsc_incorrect_answer_generation",
    "Lots-of-LoRAs/task620_ohsumed_medical_subject_headings_answer_generation",
    "Lots-of-LoRAs/task294_storycommonsense_motiv_text_generation",
    "Lots-of-LoRAs/task641_esnli_classification",
    "Lots-of-LoRAs/task318_stereoset_classification_gender",
    "Lots-of-LoRAs/task846_pubmedqa_classification",
    "Lots-of-LoRAs/task316_crows-pairs_classification_stereotype",
    "Lots-of-LoRAs/task1188_count_max_freq_char",
    "Lots-of-LoRAs/task629_dbpedia_14_classification",
    "Lots-of-LoRAs/task770_pawsx_english_text_modification",
    "Lots-of-LoRAs/task1482_gene_extraction_chemprot_dataset",
    "Lots-of-LoRAs/task499_extract_and_add_all_numbers_from_list",
    "Lots-of-LoRAs/task955_wiki_auto_style_transfer",
    "Lots-of-LoRAs/task719_mmmlu_answer_generation_management",
    "Lots-of-LoRAs/task723_mmmlu_answer_generation_moral_disputes",
    "Lots-of-LoRAs/task087_new_operator_addsub_arithmetic",
    "Lots-of-LoRAs/task211_logic2text_classification",
    "Lots-of-LoRAs/task901_freebase_qa_category_question_generation",
    "Lots-of-LoRAs/task1483_chemical_extraction_chemprot_dataset",
    "Lots-of-LoRAs/task089_swap_words_verification",
    "Lots-of-LoRAs/task627_xlwic_word_with_same_meaning_sentence_generation",
    "Lots-of-LoRAs/task153_tomqa_find_location_hard_clean",
    "Lots-of-LoRAs/task1342_amazon_us_reviews_title",
    "Lots-of-LoRAs/task828_copa_commonsense_cause_effect",
    "Lots-of-LoRAs/task064_all_elements_except_first_i",
    "Lots-of-LoRAs/task1387_anli_r3_entailment",
    "Lots-of-LoRAs/task400_paws_paraphrase_classification",
    "Lots-of-LoRAs/task1294_wiki_qa_answer_verification",
    "Lots-of-LoRAs/task243_count_elements_in_set_intersection",
    "Lots-of-LoRAs/task1572_samsum_summary",
    "Lots-of-LoRAs/task1151_swap_max_min",
    "Lots-of-LoRAs/task574_air_dialogue_sentence_generation",
    "Lots-of-LoRAs/task428_senteval_inversion",
    "Lots-of-LoRAs/task366_synthetic_return_primes",
    "Lots-of-LoRAs/task926_coached_conv_pref_word_generation",
    "Lots-of-LoRAs/task1503_hatexplain_classification",
    "Lots-of-LoRAs/task130_scan_structured_text_generation_command_action_long",
    "Lots-of-LoRAs/task515_senteval_odd_word_out",
    "Lots-of-LoRAs/task151_tomqa_find_location_easy_clean",
    "Lots-of-LoRAs/task619_ohsumed_abstract_title_generation",
    "Lots-of-LoRAs/task1562_zest_text_modification",
    "Lots-of-LoRAs/task632_dbpedia_14_classification",
    "Lots-of-LoRAs/task966_ruletaker_fact_checking_based_on_given_context",
    "Lots-of-LoRAs/task605_find_the_longest_common_subsequence_in_two_lists",
    "Lots-of-LoRAs/task1487_organism_substance_extraction_anem_dataset",
    "Lots-of-LoRAs/task707_mmmlu_answer_generation_high_school_microeconomics",
    "Lots-of-LoRAs/task1379_quarel_incorrect_answer_generation",
    "Lots-of-LoRAs/task1489_sarcasmdetection_tweet_classification",
    "Lots-of-LoRAs/task1567_propara_question_generation",
    "Lots-of-LoRAs/task1384_deal_or_no_dialog_classification",
    "Lots-of-LoRAs/task1557_jfleg_answer_generation",
    "Lots-of-LoRAs/task1404_date_conversion",
    "Lots-of-LoRAs/task691_mmmlu_answer_generation_college_physics",
    "Lots-of-LoRAs/task728_mmmlu_answer_generation_professional_accounting",
    "Lots-of-LoRAs/task219_rocstories_title_answer_generation",
    "Lots-of-LoRAs/task964_librispeech_asr_text_auto_completion",
    "Lots-of-LoRAs/task1509_evalution_antonyms",
    "Lots-of-LoRAs/task582_naturalquestion_answer_generation",
    "Lots-of-LoRAs/task455_swag_context_generation",
    "Lots-of-LoRAs/task963_librispeech_asr_next_word_prediction",
    "Lots-of-LoRAs/task382_hybridqa_answer_generation",
    "Lots-of-LoRAs/task859_prost_question_generation",
    "Lots-of-LoRAs/task1393_superglue_copa_text_completion",
    "Lots-of-LoRAs/task1565_triviaqa_classification",
    "Lots-of-LoRAs/task1720_civil_comments_toxicity_classification",
    "Lots-of-LoRAs/task670_ambigqa_question_generation",
    "Lots-of-LoRAs/task689_mmmlu_answer_generation_college_mathematics",
    "Lots-of-LoRAs/task324_jigsaw_classification_disagree",
    "Lots-of-LoRAs/task304_numeric_fused_head_resolution",
    "Lots-of-LoRAs/task1420_mathqa_general",
    "Lots-of-LoRAs/task618_amazonreview_summary_text_generation",
    "Lots-of-LoRAs/task625_xlwic_true_or_false_answer_generation",
    "Lots-of-LoRAs/task377_remove_words_of_given_length",
    "Lots-of-LoRAs/task929_products_reviews_classification",
    "Lots-of-LoRAs/task296_storycloze_correct_end_classification",
    "Lots-of-LoRAs/task852_synthetic_multiply_odds",
    "Lots-of-LoRAs/task1332_check_leap_year",
    "Lots-of-LoRAs/task1444_round_power_of_two",
    "Lots-of-LoRAs/task850_synthetic_longest_palindrome",
    "Lots-of-LoRAs/task708_mmmlu_answer_generation_high_school_physics",
    "Lots-of-LoRAs/task1292_yelp_review_full_text_categorization",
    "Lots-of-LoRAs/task110_logic2text_sentence_generation",
    "Lots-of-LoRAs/task155_count_nouns_verbs",
    "Lots-of-LoRAs/task429_senteval_tense",
    "Lots-of-LoRAs/task245_check_presence_in_set_intersection",
    "Lots-of-LoRAs/task137_detoxifying-lms_classification_toxicity",
    "Lots-of-LoRAs/task1566_propara_structured_text_generation",
    "Lots-of-LoRAs/task1146_country_capital",
    "Lots-of-LoRAs/task924_event2mind_word_generation",
    "Lots-of-LoRAs/task022_cosmosqa_passage_inappropriate_binary",
    "Lots-of-LoRAs/task118_semeval_2019_task10_open_vocabulary_mathematical_answer_generation",
    "Lots-of-LoRAs/task687_mmmlu_answer_generation_college_chemistry",
    "Lots-of-LoRAs/task1167_penn_treebank_coarse_pos_tagging",
    "Lots-of-LoRAs/task380_boolq_yes_no_question",
    "Lots-of-LoRAs/task035_winogrande_question_modification_person",
    "Lots-of-LoRAs/task033_winogrande_answer_generation",
    "Lots-of-LoRAs/task1502_hatexplain_classification",
    "Lots-of-LoRAs/task865_mawps_addsub_question_answering",
    "Lots-of-LoRAs/task181_outcome_extraction",
    "Lots-of-LoRAs/task228_arc_answer_generation_easy",
    "Lots-of-LoRAs/task698_mmmlu_answer_generation_global_facts",
    "Lots-of-LoRAs/task956_leetcode_420_strong_password_check",
    "Lots-of-LoRAs/task732_mmmlu_answer_generation_public_relations",
    "Lots-of-LoRAs/task721_mmmlu_answer_generation_medical_genetics",
    "Lots-of-LoRAs/task370_synthetic_remove_divisible_by_3",
    "Lots-of-LoRAs/task1400_obqa_incorrect_answer_generation",
    "Lots-of-LoRAs/task1199_atomic_classification_xattr",
    "Lots-of-LoRAs/task1606_ethos_text_classification",
    "Lots-of-LoRAs/task288_gigaword_summarization",
    "Lots-of-LoRAs/task1670_md_gender_bias_text_modification",
    "Lots-of-LoRAs/task207_max_element_lists",
    "Lots-of-LoRAs/task1206_atomic_classification_isbefore",
    "Lots-of-LoRAs/task457_matres_conditional_classification",
    "Lots-of-LoRAs/task1308_amazonreview_category_classification",
    "Lots-of-LoRAs/task1310_amazonreview_rating_classification",
    "Lots-of-LoRAs/task874_opus_xhosanavy_sr",
    "Lots-of-LoRAs/task1541_agnews_classification",
    "Lots-of-LoRAs/task1609_xquad_en_question_generation",
    "Lots-of-LoRAs/task210_logic2text_structured_text_generation",
    "Lots-of-LoRAs/task614_glucose_cause_event_detection",
    "Lots-of-LoRAs/task1318_country_national_dish",
    "Lots-of-LoRAs/task365_synthetic_remove_vowels",
    "Lots-of-LoRAs/task755_find_longest_substring_and_replace_its_sorted_lowercase_version_in_both_lists",
    "Lots-of-LoRAs/task123_conala_sort_dictionary",
    "Lots-of-LoRAs/task1316_remove_duplicates_string",
    "Lots-of-LoRAs/task1378_quarel_correct_answer_generation",
    "Lots-of-LoRAs/task475_yelp_polarity_classification",
    "Lots-of-LoRAs/task903_deceptive_opinion_spam_classification",
    "Lots-of-LoRAs/task070_abductivenli_incorrect_classification",
    "Lots-of-LoRAs/task720_mmmlu_answer_generation_marketing",
    "Lots-of-LoRAs/task067_abductivenli_answer_generation",
    "Lots-of-LoRAs/task1564_triviaqa_answer_generation",
    "Lots-of-LoRAs/task270_csrg_counterfactual_context_generation",
    "Lots-of-LoRAs/task167_strategyqa_question_generation",
    "Lots-of-LoRAs/task1504_hatexplain_answer_generation",
    "Lots-of-LoRAs/task178_quartz_question_answering",
    "Lots-of-LoRAs/task277_stereoset_sentence_generation_stereotype",
    "Lots-of-LoRAs/task1315_find_range_array",
    "Lots-of-LoRAs/task1434_head_qa_classification",
    "Lots-of-LoRAs/task192_hotpotqa_sentence_generation",
    "Lots-of-LoRAs/task1157_bard_analogical_reasoning_rooms_for_containers",
    "Lots-of-LoRAs/task672_nummersense",
    "Lots-of-LoRAs/task563_discofuse_answer_generation",
    "Lots-of-LoRAs/task714_mmmlu_answer_generation_human_sexuality",
    "Lots-of-LoRAs/task1212_atomic_classification_hasproperty",
    "Lots-of-LoRAs/task495_semeval_headline_classification",
    "Lots-of-LoRAs/task1583_bless_meronym_classification",
    "Lots-of-LoRAs/task753_svamp_addition_question_answering",
    "Lots-of-LoRAs/task343_winomt_classification_profession_anti",
    "Lots-of-LoRAs/task1427_country_region_in_world",
    "Lots-of-LoRAs/task092_check_prime_classification",
    "Lots-of-LoRAs/task1285_kpa_keypoint_matching",
    "Lots-of-LoRAs/task333_hateeval_classification_hate_en",
    "Lots-of-LoRAs/task329_gap_classification",
    "Lots-of-LoRAs/task398_semeval_2018_task1_tweet_joy_detection",
    "Lots-of-LoRAs/task157_count_vowels_and_consonants",
    "Lots-of-LoRAs/task074_squad1.1_question_generation",
    "Lots-of-LoRAs/task1506_celebrity_minimal_dob_span",
    "Lots-of-LoRAs/task697_mmmlu_answer_generation_formal_logic",
    "Lots-of-LoRAs/task285_imdb_answer_generation",
    "Lots-of-LoRAs/task393_plausible_result_generation",
    "Lots-of-LoRAs/task1147_country_currency",
    "Lots-of-LoRAs/task1585_root09_hypernym_generation",
    "Lots-of-LoRAs/task648_answer_generation",
    "Lots-of-LoRAs/task353_casino_classification_negotiation_elicit_pref",
    "Lots-of-LoRAs/task1431_head_qa_answer_generation",
    "Lots-of-LoRAs/task148_afs_argument_quality_gay_marriage",
    "Lots-of-LoRAs/task585_preposition_classification",
    "Lots-of-LoRAs/task081_piqa_wrong_answer_generation",
    "Lots-of-LoRAs/task477_cls_english_dvd_classification",
    "Lots-of-LoRAs/task1582_bless_hypernym_generation",
    "Lots-of-LoRAs/task355_casino_classification_negotiation_other_need",
    "Lots-of-LoRAs/task381_boolq_question_generation",
    "Lots-of-LoRAs/task633_dbpedia_14_answer_generation",
    "Lots-of-LoRAs/task093_conala_normalize_lists",
    "Lots-of-LoRAs/task722_mmmlu_answer_generation_random_topic",
    "Lots-of-LoRAs/task566_circa_classification",
    "Lots-of-LoRAs/task1152_bard_analogical_reasoning_causation",
    "Lots-of-LoRAs/task1452_location_entity_extraction_btc_corpus",
    "Lots-of-LoRAs/task1286_openbookqa_question_answering",
    "Lots-of-LoRAs/task925_coached_conv_pref_classifier",
    "Lots-of-LoRAs/task1703_ljspeech_textmodification",
    "Lots-of-LoRAs/task833_poem_sentiment_classification",
    "Lots-of-LoRAs/task1210_atomic_classification_madeupof",
    "Lots-of-LoRAs/task679_hope_edi_english_text_classification",
    "Lots-of-LoRAs/task1203_atomic_classification_xreact",
    "Lots-of-LoRAs/task1089_check_monotonic_array",
    "Lots-of-LoRAs/task389_torque_generate_temporal_question",
    "Lots-of-LoRAs/task378_reverse_words_of_given_length",
    "Lots-of-LoRAs/task637_extract_and_sort_unique_digits_in_a_list",
    "Lots-of-LoRAs/task101_reverse_and_concatenate_all_elements_from_index_i_to_j",
    "Lots-of-LoRAs/task1355_sent_comp_summarization",
    "Lots-of-LoRAs/task640_esnli_classification",
    "Lots-of-LoRAs/task344_hybridqa_answer_generation",
    "Lots-of-LoRAs/task701_mmmlu_answer_generation_high_school_computer_science",
    "Lots-of-LoRAs/task1190_add_integer_to_list",
    "Lots-of-LoRAs/task733_mmmlu_answer_generation_security_studies",
    "Lots-of-LoRAs/task107_splash_question_to_sql",
    "Lots-of-LoRAs/task1209_atomic_classification_objectuse",
    "Lots-of-LoRAs/task505_count_all_numerical_elements_in_list",
    "Lots-of-LoRAs/task1385_anli_r1_entailment",
    "Lots-of-LoRAs/task1135_xcsr_en_commonsense_mc_classification",
    "Lots-of-LoRAs/task328_jigsaw_classification_insult",
    "Lots-of-LoRAs/task609_sbic_potentially_offense_binary_classification",
    "Lots-of-LoRAs/task413_mickey_en_sentence_perturbation_generation",
    "Lots-of-LoRAs/task072_abductivenli_answer_generation",
    "Lots-of-LoRAs/task1425_country_iso_numeric",
    "Lots-of-LoRAs/task1451_drug_dose_extraction",
    "Lots-of-LoRAs/task713_mmmlu_answer_generation_human_aging",
    "Lots-of-LoRAs/task642_esnli_classification",
    "Lots-of-LoRAs/task1321_country_continent",
    "Lots-of-LoRAs/task454_swag_incorrect_answer_generation",
    "Lots-of-LoRAs/task504_count_all_alphabetical_elements_in_list",
    "Lots-of-LoRAs/task696_mmmlu_answer_generation_elementary_mathematics",
    "Lots-of-LoRAs/task1429_evalution_semantic_relation_classification",
    "Lots-of-LoRAs/task1645_medical_question_pair_dataset_text_classification",
    "Lots-of-LoRAs/task431_senteval_object_count",
    "Lots-of-LoRAs/task1317_country_calling_code",
    "Lots-of-LoRAs/task131_scan_long_text_generation_action_command_long",
    "Lots-of-LoRAs/task675_google_wellformed_query_sentence_generation",
    "Lots-of-LoRAs/task1158_bard_analogical_reasoning_manipulating_items",
    "Lots-of-LoRAs/task1325_qa_zre_question_generation_on_subject_relation",
    "Lots-of-LoRAs/task1216_atomic_classification_causes",
    "Lots-of-LoRAs/task1347_glue_sts-b_similarity_classification",
    "Lots-of-LoRAs/task1328_qa_zre_relation_generation_from_question",
    "Lots-of-LoRAs/task630_dbpedia_14_classification",
    "Lots-of-LoRAs/task460_qasper_answer_generation",
    "Lots-of-LoRAs/task1665_trainglecopa_question_generation",
    "Lots-of-LoRAs/task1154_bard_analogical_reasoning_travel",
    "Lots-of-LoRAs/task300_storycloze_order_generation",
    "Lots-of-LoRAs/task617_amazonreview_category_text_generation",
    "Lots-of-LoRAs/task1508_wordnet_antonyms",
    "Lots-of-LoRAs/task628_xlwic_word_with_different_meaning_sentence_generation",
    "Lots-of-LoRAs/task600_find_the_longest_common_substring_in_two_strings",
    "Lots-of-LoRAs/task1380_quarel_correct_option_generation",
    "Lots-of-LoRAs/task489_mwsc_question_generation",
    "Lots-of-LoRAs/task905_hate_speech_offensive_classification",
    "Lots-of-LoRAs/task065_timetravel_consistent_sentence_classification",
    "Lots-of-LoRAs/task044_essential_terms_identifying_essential_words",
    "Lots-of-LoRAs/task069_abductivenli_classification",
    "Lots-of-LoRAs/task356_casino_classification_negotiation_self_need",
    "Lots-of-LoRAs/task403_creak_commonsense_inference",
    "Lots-of-LoRAs/task590_amazonfood_summary_correction_classification",
    "Lots-of-LoRAs/task577_curiosity_dialogs_classification",
    "Lots-of-LoRAs/task819_pec_sentiment_classification",
    "Lots-of-LoRAs/task1721_civil_comments_obscenity_classification",
    "Lots-of-LoRAs/task351_winomt_classification_gender_identifiability_anti",
    "Lots-of-LoRAs/task904_hate_speech_offensive_classification",
    "Lots-of-LoRAs/task244_count_elements_in_set_union",
    "Lots-of-LoRAs/task1712_poki_classification",
    "Lots-of-LoRAs/task683_online_privacy_policy_text_purpose_answer_generation",
    "Lots-of-LoRAs/task1197_atomic_classification_oreact",
    "Lots-of-LoRAs/task1592_yahoo_answers_topics_classfication",
    "Lots-of-LoRAs/task274_overruling_legal_classification",
    "Lots-of-LoRAs/task278_stereoset_sentence_generation_antistereotype",
    "Lots-of-LoRAs/task1534_daily_dialog_question_classification",
    "Lots-of-LoRAs/task891_gap_coreference_resolution",
    "Lots-of-LoRAs/task694_mmmlu_answer_generation_econometrics",
    "Lots-of-LoRAs/task497_extract_all_numbers_from_list_in_order",
    "Lots-of-LoRAs/task488_extract_all_alphabetical_elements_from_list_in_order",
    "Lots-of-LoRAs/task144_subjqa_question_answering",
    "Lots-of-LoRAs/task1722_civil_comments_threat_classification",
    "Lots-of-LoRAs/task1728_web_nlg_data_to_text",
    "Lots-of-LoRAs/task291_semeval_2020_task4_commonsense_validation",
    "Lots-of-LoRAs/task284_imdb_classification",
    "Lots-of-LoRAs/task1288_glue_mrpc_paraphrasing",
    "Lots-of-LoRAs/task128_scan_structured_text_generation_command_action_short",
    "Lots-of-LoRAs/task580_socialiqa_answer_generation",
    "Lots-of-LoRAs/task616_cola_classification",
    "Lots-of-LoRAs/task1401_obqa_sentence_generation",
    "Lots-of-LoRAs/task716_mmmlu_answer_generation_jurisprudence",
    "Lots-of-LoRAs/task1421_mathqa_other",
    "Lots-of-LoRAs/task1656_gooaq_answer_generation",
    "Lots-of-LoRAs/task1311_amazonreview_rating_classification",
    "Lots-of-LoRAs/task516_senteval_conjoints_inversion",
    "Lots-of-LoRAs/task593_sciq_explanation_generation",
    "Lots-of-LoRAs/task138_detoxifying-lms_classification_fluency",
    "Lots-of-LoRAs/task1186_nne_hrngo_classification",
    "Lots-of-LoRAs/task362_spolin_yesand_prompt_response_sub_classification",
    "Lots-of-LoRAs/task1326_qa_zre_question_generation_from_answer",
    "Lots-of-LoRAs/task119_semeval_2019_task10_geometric_mathematical_answer_generation",
    "Lots-of-LoRAs/task108_contextualabusedetection_classification",
    "Lots-of-LoRAs/task584_udeps_eng_fine_pos_tagging",
    "Lots-of-LoRAs/task388_torque_token_classification",
    "Lots-of-LoRAs/task045_miscellaneous_sentence_paraphrasing",
    "Lots-of-LoRAs/task821_protoqa_question_generation",
    "Lots-of-LoRAs/task1581_eqasc-perturbed_answer_generation",
    "Lots-of-LoRAs/task695_mmmlu_answer_generation_electrical_engineering",
    "Lots-of-LoRAs/task596_mocha_question_generation",
    "Lots-of-LoRAs/task568_circa_question_generation",
    "Lots-of-LoRAs/task085_unnatural_addsub_arithmetic",
    "Lots-of-LoRAs/task1495_adverse_drug_event_classification",
    "Lots-of-LoRAs/task927_yelp_negative_to_positive_style_transfer",
    "Lots-of-LoRAs/task1453_person_entity_extraction_btc_corpus",
    "Lots-of-LoRAs/task1201_atomic_classification_xintent",
    "Lots-of-LoRAs/task923_event2mind_classifier",
    "Lots-of-LoRAs/task1204_atomic_classification_hinderedby",
    "Lots-of-LoRAs/task1510_evalution_relation_extraction",
    "Lots-of-LoRAs/task754_svamp_common-division_question_answering",
    "Lots-of-LoRAs/task1403_check_validity_date_mmddyyyy",
    "Lots-of-LoRAs/task1192_food_flavor_profile",
    "Lots-of-LoRAs/task565_circa_answer_generation",
    "Lots-of-LoRAs/task146_afs_argument_similarity_gun_control",
    "Lots-of-LoRAs/task666_mmmlu_answer_generation_astronomy",
    "Lots-of-LoRAs/task050_multirc_answerability",
    "Lots-of-LoRAs/task704_mmmlu_answer_generation_high_school_government_and_politics",
    "Lots-of-LoRAs/task934_turk_simplification",
    "Lots-of-LoRAs/task579_socialiqa_classification",
    "Lots-of-LoRAs/task1196_atomic_classification_oeffect",
    "Lots-of-LoRAs/task267_concatenate_and_reverse_all_elements_from_index_i_to_j",
    "Lots-of-LoRAs/task206_collatz_conjecture",
    "Lots-of-LoRAs/task936_defeasible_nli_snli_classification",
    "Lots-of-LoRAs/task323_jigsaw_classification_sexually_explicit",
    "Lots-of-LoRAs/task494_review_polarity_answer_generation",
    "Lots-of-LoRAs/task461_qasper_question_generation",
    "Lots-of-LoRAs/task1409_dart_text_generation",
    "Lots-of-LoRAs/task1313_amazonreview_polarity_classification",
    "Lots-of-LoRAs/task076_splash_correcting_sql_mistake",
    "Lots-of-LoRAs/task686_mmmlu_answer_generation_college_biology",
    "Lots-of-LoRAs/task740_lhoestq_answer_generation_quantity",
    "Lots-of-LoRAs/task034_winogrande_question_modification_object",
    "Lots-of-LoRAs/task1211_atomic_classification_hassubevent",
    "Lots-of-LoRAs/task113_count_frequency_of_letter",
    "Lots-of-LoRAs/task280_stereoset_classification_stereotype_type",
    "Lots-of-LoRAs/task1551_every_ith_element_from_kth_element",
    "Lots-of-LoRAs/task116_com2sense_commonsense_reasoning",
    "Lots-of-LoRAs/task518_emo_different_dialogue_emotions",
    "Lots-of-LoRAs/task1520_qa_srl_answer_generation",
    "Lots-of-LoRAs/task079_conala_concat_strings",
    "Lots-of-LoRAs/task513_argument_stance_classification",
    "Lots-of-LoRAs/task1590_diplomacy_text_generation",
    "Lots-of-LoRAs/task1713_convai3_sentence_generation",
    "Lots-of-LoRAs/task1386_anli_r2_entailment",
    "Lots-of-LoRAs/task063_first_i_elements",
    "Lots-of-LoRAs/task183_rhyme_generation",
    "Lots-of-LoRAs/task1447_drug_extraction_ade",
    "Lots-of-LoRAs/task671_ambigqa_text_generation",
    "Lots-of-LoRAs/task068_abductivenli_incorrect_answer_generation",
    "Lots-of-LoRAs/task858_inquisitive_span_detection",
    "Lots-of-LoRAs/task699_mmmlu_answer_generation_high_school_biology",
    "Lots-of-LoRAs/task1593_yahoo_answers_topics_classification",
    "Lots-of-LoRAs/task700_mmmlu_answer_generation_high_school_chemistry",
    "Lots-of-LoRAs/task1607_ethos_text_classification",
    "Lots-of-LoRAs/task121_zest_text_modification",
    "Lots-of-LoRAs/task190_snli_classification",
    "Lots-of-LoRAs/task1168_brown_coarse_pos_tagging",
    "Lots-of-LoRAs/task195_sentiment140_classification",
    "Lots-of-LoRAs/task1723_civil_comments_sexuallyexplicit_classification",
    "Lots-of-LoRAs/task1449_disease_entity_extraction_bc5cdr_dataset",
    "Lots-of-LoRAs/task363_sst2_polarity_classification",
    "Lots-of-LoRAs/task1419_mathqa_gain",
    "Lots-of-LoRAs/task1398_obqa_question_generation",
    "Lots-of-LoRAs/task893_gap_fill_the_blank_coreference_resolution",
    "Lots-of-LoRAs/task326_jigsaw_classification_obscene",
    "Lots-of-LoRAs/task1194_kth_largest_element",
    "Lots-of-LoRAs/task102_commongen_sentence_generation",
    "Lots-of-LoRAs/task145_afs_argument_similarity_death_penalty",
    "Lots-of-LoRAs/task1338_peixian_equity_evaluation_corpus_sentiment_classifier",
    "Lots-of-LoRAs/task391_causal_relationship",
    "Lots-of-LoRAs/task176_break_decompose_questions",
    "Lots-of-LoRAs/task319_stereoset_classification_profession",
    "Lots-of-LoRAs/task359_casino_classification_negotiation_vouch_fair",
    "Lots-of-LoRAs/task856_conv_ai_2_classification",
    "Lots-of-LoRAs/task1729_personachat_generate_next",
    "Lots-of-LoRAs/task761_app_review_classification",
    "Lots-of-LoRAs/task1320_country_domain_tld",
    "Lots-of-LoRAs/task1596_event2mind_text_generation_2",
    "Lots-of-LoRAs/task1601_webquestions_answer_generation",
    "Lots-of-LoRAs/task615_moviesqa_answer_generation",
    "Lots-of-LoRAs/task706_mmmlu_answer_generation_high_school_mathematics",
    "Lots-of-LoRAs/task1283_hrngo_quality_classification",
    "Lots-of-LoRAs/task638_multi_woz_classification",
    "Lots-of-LoRAs/task607_sbic_intentional_offense_binary_classification",
    "Lots-of-LoRAs/task692_mmmlu_answer_generation_computer_security",
    "Lots-of-LoRAs/task588_amazonfood_rating_classification",
    "Lots-of-LoRAs/task129_scan_long_text_generation_action_command_short",
    "Lots-of-LoRAs/task1200_atomic_classification_xeffect",
    "Lots-of-LoRAs/task1486_cell_extraction_anem_dataset",
    "Lots-of-LoRAs/task1406_kth_smallest_element",
    "Lots-of-LoRAs/task583_udeps_eng_coarse_pos_tagging",
    "Lots-of-LoRAs/task094_conala_calculate_mean",
    "Lots-of-LoRAs/task1731_quartz_question_answering",
    "Lots-of-LoRAs/task308_jeopardy_answer_generation_all",
    "Lots-of-LoRAs/task664_mmmlu_answer_generation_abstract_algebra",
    "Lots-of-LoRAs/task163_count_words_ending_with_letter",
    "Lots-of-LoRAs/task127_scan_long_text_generation_action_command_all",
    "Lots-of-LoRAs/task736_mmmlu_answer_generation_virology",
    "Lots-of-LoRAs/task379_agnews_topic_classification",
    "Lots-of-LoRAs/task330_gap_answer_generation",
    "Lots-of-LoRAs/task1599_smcalflow_classification",
    "Lots-of-LoRAs/task875_emotion_classification",
    "Lots-of-LoRAs/task1214_atomic_classification_xwant",
    "Lots-of-LoRAs/task933_wiki_auto_style_transfer",
    "Lots-of-LoRAs/task1319_country_by_barcode_prefix",
    "Lots-of-LoRAs/task456_matres_intention_classification",
    "Lots-of-LoRAs/task1189_check_char_in_string",
    "Lots-of-LoRAs/task1657_gooaq_question_generation",
    "Lots-of-LoRAs/task710_mmmlu_answer_generation_high_school_statistics",
    "Lots-of-LoRAs/task517_emo_classify_emotion_of_dialogue",
    "Lots-of-LoRAs/task385_socialiqa_incorrect_answer_generation",
    "Lots-of-LoRAs/task472_haspart_classification",
    "Lots-of-LoRAs/task047_miscellaneous_answering_science_questions",
    "Lots-of-LoRAs/task1533_daily_dialog_formal_classification",
    "Lots-of-LoRAs/task249_enhanced_wsc_pronoun_disambiguation",
    "Lots-of-LoRAs/task095_conala_max_absolute_value",
    "Lots-of-LoRAs/task1479_organization_entity_extraction_btc_corpus",
    "Lots-of-LoRAs/task724_mmmlu_answer_generation_moral_scenarios",
    "Lots-of-LoRAs/task726_mmmlu_answer_generation_philosophy",
    "Lots-of-LoRAs/task1418_bless_semantic_relation_classification",
    "Lots-of-LoRAs/task507_position_of_all_numerical_elements_in_list",
    "Lots-of-LoRAs/task043_essential_terms_answering_incomplete_questions",
    "Lots-of-LoRAs/task1394_meta_woz_task_classification",
    "Lots-of-LoRAs/task149_afs_argument_quality_death_penalty",
    "Lots-of-LoRAs/task1156_bard_analogical_reasoning_tools",
    "Lots-of-LoRAs/task1598_nyc_long_text_generation",
    "Lots-of-LoRAs/task322_jigsaw_classification_threat",
    "Lots-of-LoRAs/task1714_convai3_sentence_generation",
    "Lots-of-LoRAs/task727_mmmlu_answer_generation_prehistory",
    "Lots-of-LoRAs/task080_piqa_answer_generation",
    "Lots-of-LoRAs/task1704_ljspeech_textmodification",
    "Lots-of-LoRAs/task077_splash_explanation_to_sql",
    "Lots-of-LoRAs/task1390_wscfixed_coreference",
    "Lots-of-LoRAs/task667_mmmlu_answer_generation_business_ethics",
    "Lots-of-LoRAs/task1724_civil_comments_insult_classification",
    "Lots-of-LoRAs/task1088_array_of_products",
    "Lots-of-LoRAs/task1087_two_number_sum",
    "Lots-of-LoRAs/task550_discofuse_sentence_generation",
    "Lots-of-LoRAs/task892_gap_reverse_coreference_resolution",
    "Lots-of-LoRAs/task674_google_wellformed_query_sentence_generation",
    "Lots-of-LoRAs/task1389_hellaswag_completion",
    "Lots-of-LoRAs/task509_collate_of_all_alphabetical_and_numerical_elements_in_list_separately",
    "Lots-of-LoRAs/task346_hybridqa_classification",
    "Lots-of-LoRAs/task769_qed_summarization",
    "Lots-of-LoRAs/task1391_winogrande_easy_answer_generation",
    "Lots-of-LoRAs/task1322_country_government_type",
    "Lots-of-LoRAs/task341_winomt_classification_gender_anti",
    "Lots-of-LoRAs/task290_tellmewhy_question_answerability",
    "Lots-of-LoRAs/task335_hateeval_classification_aggresive_en",
    "Lots-of-LoRAs/task879_schema_guided_dstc8_classification",
    "Lots-of-LoRAs/task861_asdiv_addsub_question_answering",
    "Lots-of-LoRAs/task594_sciq_question_generation",
    "Lots-of-LoRAs/task066_timetravel_binary_consistency_classification",
    "Lots-of-LoRAs/task162_count_words_starting_with_letter",
    "Lots-of-LoRAs/task1584_evalution_meronym_classification",
    "Lots-of-LoRAs/task1622_disfl_qa_text_modication",
    "Lots-of-LoRAs/task247_dream_answer_generation",
]

LOL_DATASET_NAMES = {
    "lol_" + path.split("Lots-of-LoRAs/task")[1].split("_")[0]: path
    for path in lol_paths
}


@torch.no_grad()
def eval(
    model_dir,
    lora_dirs,
    task,
    chat_template,
    gpu_memory_utilization=0.7,
    ds_kwargs=None,
    use_icl=False,
    use_task_desc=False,
    per_sample_lora=True,
    pre_created_model=None,
    max_context_length: Optional[int] = None,
    rope_scaling: Optional[dict[str, Any]] = None,
    retrieval_k: Optional[int] = None,
):
    timing_info: dict[str, Any] = {
        "per_lora_seconds": {},
        "per_call_seconds": [],
    }
    wall_clock_start = time.perf_counter()
    timing_info["started_at"] = time.time()

    def _record_time(lora_key: str | None, seconds: float) -> None:
        if lora_key is None:
            return
        aggregate = timing_info["per_lora_seconds"].get(lora_key, 0.0) + float(seconds)
        timing_info["per_lora_seconds"][lora_key] = aggregate
        timing_info["per_call_seconds"].append({"lora_dir": lora_key, "seconds": float(seconds)})

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    
    in_context_message = ""
    if use_icl and task in IN_CONTEXT_EXAMPLES:
        in_context_message = IN_CONTEXT_EXAMPLES[task]
    elif use_task_desc and task in TASK_DESCRIPTION_MESSAGES:
        in_context_message = TASK_DESCRIPTION_MESSAGES[task]

    # Check if it's a task that uses the new config-based approach
    if any(x in task for x in ["opinionqa", "lamp_", "longlamp_", "personalreddit", "prism", "EC_", "aloe_"]):
        prefill_text = None
        if use_icl and task not in ["gsm8k", "humaneval", "mbpp"]:
            prefill_text = "Answer:"

        return eval_task_from_config(
            model_dir=model_dir,
            task_name=task,
            lora_dirs=lora_dirs,
            chat_template=chat_template,
            gpu_memory_utilization=gpu_memory_utilization,
            prefill_text=prefill_text,
            per_sample_lora=per_sample_lora,
            pre_created_model=pre_created_model,
            in_context_message=in_context_message,
            retrieval_k=retrieval_k,
            max_context_length=max_context_length,
            rope_scaling=rope_scaling,
        )

    # Legacy handling for LOL tasks
    if task.startswith("lol"):
        dataset_name = LOL_DATASET_NAMES[task]
        return eval_rouge(
            model_dir,
            lora_dirs,
            chat_template,
            gpu_memory_utilization,
            system_message="",
            template=LOL_TEMPLATE,
            dataset_name=dataset_name,
            dataset_kwargs=dict(split="test"),
            response_field="answer",
            prefill_text="",
            preprocessing_fn=get_preprocessing_fn(task),
            pre_created_model=pre_created_model,
        )
    
    # Legacy handling for other tasks
    eval_kwargs = dict(
        model_dir=model_dir,
        lora_dirs=lora_dirs,
        chat_template=chat_template,
        in_context_message=in_context_message,
        gpu_memory_utilization=gpu_memory_utilization,
        per_sample_lora=per_sample_lora,
        pre_created_model=pre_created_model,
        max_context_length=max_context_length,
        rope_scaling=rope_scaling,
    )
    if use_icl and task not in ["gsm8k", "humaneval", "mbpp"]:
        # LLMs tend to copy the pattern in the in-context examples
        # so we prefill "Answer:" pattern
        eval_kwargs["prefill_text"] = "Answer:"

    if (ds_kwargs is not None) and (task not in ["gsm8k", "humaneval", "mbpp"]):
        eval_kwargs["dataset_kwargs"] = ds_kwargs

    return EVAL_FNS[task](**eval_kwargs)
