"""
Data loading and processing for hypernetwork training.

LOCAL DATASET SUPPORT:
This module now supports loading local datasets in addition to HuggingFace Hub datasets:

1. Local HuggingFace datasets: Use load_from_disk for datasets saved with save_to_disk()
   Example in metadata YAML:
   ds_kwargs:
     path: "/path/to/local/huggingface/dataset"
     split: "train[:1000]"

2. Local JSONL files: Automatically detected by .jsonl or .jsonlines extension
   Example in metadata YAML:
   ds_kwargs:
     path: "/path/to/dataset.jsonl"
     split: "train[:500]"

3. Local JSON files: Automatically detected by .json extension
   Example in metadata YAML:
   ds_kwargs:
     path: "/path/to/dataset.json"
     split: "train[90%:]"

4. Regular HuggingFace Hub datasets: Fallback to standard load_dataset behavior
   Example in metadata YAML:
   ds_kwargs:
     path: "microsoft/orca-math-word-problems-200k"
     split: "train[:1000]"

PROFILE_TEXT MODE:
This module supports two modes for hypernetwork input:

1. Traditional mode: Uses task descriptions from metadata YAML files
2. Profile_text mode: Uses profile_text field from individual data points

When datasets contain the following fields, profile_text mode is automatically enabled:
- "user_id": Unique identifier for each user
- "question_id": Unique identifier for each question (unique across all data points)  
- "profile_text": List of profile text strings for each data point
- "input": The input text
- "output": The expected output text

In profile_text mode:
- Each data point has a "profile_text" field containing a single profile text string
- The profile text is embedded and used as hypernetwork input
- Traditional task descriptions from metadata are ignored
- All unique profile texts are pre-embedded for efficiency

Format example:
{
    "user_id": "user_001",
    "question_id": "q_001", 
    "profile_text": "I prefer step-by-step solutions",
    "input": "What is 2+2?",
    "output": "2+2=4"
}
"""

from collections import defaultdict
from functools import partial
from glob import glob
from math import ceil
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Union

import torch
import datasets
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, ConcatDataset, Sampler, DataLoader

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from hyper_llm_modulator.utils import (
    embed_texts,
    get_inp_tokenize_fn,
    get_preprocessing_fn,
    get_prompt_formatting_fn,
    repeat_iterator,
)

logger = logging.getLogger()

DATA_DIR = "data"
TRANSFORMED_DS_DIR = "data/transformed_datasets"
EMBS_DIR = "data/embs"
BENCHMARK_TASK_INFO = {
    # "openbookqa": {"split": "validation[:500]"},
    # "hellaswag": {"split": "train[:500]"},
    # "winogrande": {"name": "winogrande_debiased", "split": "train[:500]", "trust_remote_code": True},
    # "boolq": {"split": "train[:500]"},
    # "piqa": {"split": "train[:500]"},
    # "arc_easy": {"name": "ARC-Easy", "split": "validation[:500]"},
    # "arc_challenge": {"name": "ARC-Challenge", "split": "validation[:500]"},
    "lamp_citation_ood_test": {"name": "lamp_citation_ood_test"},
    "lamp_news_headline_ood_test": {"name": "lamp_news_headline_ood_test"},
    "lamp_news_cat_ood_test": {"name": "lamp_news_cat_ood_test"},
    "lamp_movie_ood_test": {"name": "lamp_movie_ood_test"},
    "lamp_product_ood_test": {"name": "lamp_product_ood_test"},
    "lamp_scholarly_title_ood_test": {"name": "lamp_scholarly_title_ood_test"},
    "lamp_tweet_ood_test": {"name": "lamp_tweet_ood_test"},
    "longlamp_abstract_generation_ood_test": {"name": "longlamp_abstract_generation_ood_test"},
    "longlamp_product_review_ood_test": {"name": "longlamp_product_review_ood_test"},
    "longlamp_topic_writing_ood_test": {"name": "longlamp_topic_writing_ood_test"},
    "prism_ood_test": {"name": "prism_ood_test"},
    "aloe_ood_test": {"name": "aloe_ood_test"},
    "EC_ood_test": {"name": "EC_ood_test"},
    "personalreddit_ood_test": {"name": "personalreddit_ood_test"},
    "lamp_citation_random_test": {"name": "lamp_citation_random_test"},
    "lamp_news_headline_random_test": {"name": "lamp_news_headline_random_test"},
    "lamp_news_cat_random_test": {"name": "lamp_news_cat_random_test"},
    "lamp_movie_random_test": {"name": "lamp_movie_random_test"},
    "lamp_product_random_test": {"name": "lamp_product_random_test"},
    "lamp_scholarly_title_random_test": {"name": "lamp_scholarly_title_random_test"},
    "lamp_tweet_random_test": {"name": "lamp_tweet_random_test"},
    "longlamp_abstract_generation_random_test": {"name": "longlamp_abstract_generation_random_test"},
    "longlamp_product_review_random_test": {"name": "longlamp_product_review_random_test"},
    "longlamp_topic_writing_random_test": {"name": "longlamp_topic_writing_random_test"},
    "prism_random_test": {"name": "prism_random_test"},
    "aloe_random_test": {"name": "aloe_random_test"},
    "EC_random_test": {"name": "EC_random_test"},
    "personalreddit_random_test": {"name": "personalreddit_random_test"},
}

# Default location for per-user embedding artifacts used when clustering training users.
DEFAULT_TRAIN_USER_EMBEDDINGS_DIR = "./data_p13n/user_gen_profile_embeddings_task_specific"


from typing import Optional


def _update_hash_map(dir_path: str, hashed_name: str, original_name: str, extra: Optional[dict] = None):
    """Persist a mapping from a hashed filename to its original name/context.

    Writes/updates a JSON file named `hash_map.json` in `dir_path` with entries like:
        { "<hashed_name>": { "original": "<original_name>", ...extra } }
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        map_fp = os.path.join(dir_path, "hash_map.json")
        mapping = {}
        if os.path.exists(map_fp):
            try:
                with open(map_fp, "r", encoding="utf-8") as f:
                    mapping = json.load(f) or {}
            except Exception:
                # Corrupt or empty map file; start fresh
                mapping = {}

        entry = {"original": original_name}
        if extra:
            # Ensure JSON-serializable values only
            safe_extra = {}
            for k, v in extra.items():
                try:
                    json.dumps(v)
                    safe_extra[k] = v
                except Exception:
                    safe_extra[k] = str(v)
            entry.update(safe_extra)

        mapping[str(hashed_name)] = entry

        with open(map_fp, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception as e:
        logger.warning(f"Failed to update hash map for {hashed_name}: {e}")


def detect_local_dataset_type(path):
    """
    Detect the type of local dataset based on the path.
    
    Returns:
        - 'huggingface_disk' if it's a local HuggingFace dataset directory
        - 'jsonl' if it's a JSONL file
        - 'json' if it's a JSON file
        - None if it's not a recognized local dataset format
    """
    if not os.path.exists(path):
        return None
    
    if os.path.isdir(path):
        # Check if it's a HuggingFace dataset directory
        # HuggingFace datasets have dataset_info.json and state.json files
        if (os.path.exists(os.path.join(path, "dataset_info.json")) and 
            os.path.exists(os.path.join(path, "state.json"))):
            return 'huggingface_disk'
    elif os.path.isfile(path):
        # Check file extension
        if path.endswith('.jsonl') or path.endswith('.jsonlines'):
            return 'jsonl'
        elif path.endswith('.json'):
            return 'json'
    
    return None


def load_local_dataset(path, split=None, **kwargs):
    """
    Load a local dataset based on its type.
    
    Args:
        path: Path to the local dataset
        split: Dataset split to load (for HuggingFace datasets)
        **kwargs: Additional arguments passed to the loading function
    
    Returns:
        Loaded dataset
    """
    dataset_type = detect_local_dataset_type(path)
    
    if dataset_type == 'huggingface_disk':
        logger.info(f"Loading HuggingFace dataset from disk: {path}")
        dataset = load_from_disk(path)
        # If split is specified and dataset is a DatasetDict, get the specific split
        if split and isinstance(dataset, datasets.DatasetDict):
            # Handle split syntax like "train[:500]" or "train[90%:]"
            if '[' in split and ']' in split:
                split_name = split.split('[')[0]
                split_slice = split[split.find('['):split.find(']')+1]
                if split_name in dataset:
                    dataset = dataset[split_name]
                    # Apply slicing if specified
                    if split_slice != '[]':
                        # Parse the slice (simplified parsing)
                        slice_content = split_slice[1:-1]  # Remove [ and ]
                        if ':' in slice_content:
                            parts = slice_content.split(':')
                            if len(parts) == 2:
                                start, end = parts
                                if start == '' and end.endswith('%'):
                                    # Handle percentage slicing like "[90%:]"
                                    pct = int(end[:-1])
                                    start_idx = int(len(dataset) * pct / 100)
                                    dataset = dataset.select(range(start_idx, len(dataset)))
                                elif start.endswith('%') and end == '':
                                    # Handle percentage slicing like "[:90%]"
                                    pct = int(start[:-1])
                                    end_idx = int(len(dataset) * pct / 100)
                                    dataset = dataset.select(range(0, end_idx))
                                elif end.isdigit():
                                    # Handle numeric slicing like "[:500]"
                                    end_idx = min(int(end), len(dataset))
                                    start_idx = int(start) if start.isdigit() else 0
                                    dataset = dataset.select(range(start_idx, end_idx))
                else:
                    logger.warning(f"Split '{split_name}' not found in dataset. Available splits: {list(dataset.keys())}")
                    # Use the first available split
                    dataset = list(dataset.values())[0]
            else:
                # Simple split name without slicing
                if split in dataset:
                    dataset = dataset[split]
                else:
                    logger.warning(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
                    dataset = list(dataset.values())[0]
        elif isinstance(dataset, datasets.DatasetDict):
            # If no split specified but it's a DatasetDict, use the first split
            dataset = list(dataset.values())[0]
        return dataset
    
    elif dataset_type in ['jsonl', 'json']:
        logger.info(f"Loading {dataset_type.upper()} dataset: {path}")
        # Use datasets library to load JSON/JSONL files
        dataset = load_dataset('json', data_files=path, split='train', **kwargs)
        
        # Handle split slicing for JSON/JSONL files
        if split and split != 'train':
            if '[' in split and ']' in split:
                split_slice = split[split.find('['):split.find(']')+1]
                if split_slice != '[]':
                    slice_content = split_slice[1:-1]  # Remove [ and ]
                    if ':' in slice_content:
                        parts = slice_content.split(':')
                        if len(parts) == 2:
                            start, end = parts
                            if start == '' and end.endswith('%'):
                                # Handle percentage slicing like "[90%:]"
                                pct = int(end[:-1])
                                start_idx = int(len(dataset) * pct / 100)
                                dataset = dataset.select(range(start_idx, len(dataset)))
                            elif start.endswith('%') and end == '':
                                # Handle percentage slicing like "[:90%]"
                                pct = int(start[:-1])
                                end_idx = int(len(dataset) * pct / 100)
                                dataset = dataset.select(range(0, end_idx))
                            elif end.isdigit():
                                # Handle numeric slicing like "[:500]"
                                end_idx = min(int(end), len(dataset))
                                start_idx = int(start) if start.isdigit() else 0
                                dataset = dataset.select(range(start_idx, end_idx))
        
        return dataset
    
    else:
        raise ValueError(f"Unsupported local dataset format at path: {path}")


def expand_dataset_for_retrieval_k(dataset, metadata_config):
    """
    Expand dataset based on retrieval_k parameter in metadata.
    For each retrieval_k value, create a copy of each example with the appropriate
    profile_retrieval_k{k} field mapped to profile_retrieval_k.
    """
    retrieval_k_values = metadata_config.get('retrieval_k', [1])
    
    if len(retrieval_k_values) == 1 and retrieval_k_values[0] == 1:
        # No expansion needed, just map profile_retrieval_k1 to profile_retrieval_k
        def map_k1_to_generic(example):
            if 'profile_retrieval_k1' in example:
                example['profile_retrieval_k'] = example['profile_retrieval_k1']
            return example
        return dataset.map(map_k1_to_generic, batched=False)
    
    # Expand dataset for multiple k values
    expanded_examples = []
    
    for example in dataset:
        for k in retrieval_k_values:
            # Create a copy of the example for this k value
            expanded_example = dict(example)
            
            # Map the appropriate profile_retrieval_k{k} field to profile_retrieval_k
            k_field = f'profile_retrieval_k{k}'
            if k_field in example:
                expanded_example['profile_retrieval_k'] = example[k_field]
            else:
                # Fallback to empty string if the specific k field doesn't exist
                expanded_example['profile_retrieval_k'] = ""
            
            # Add a field to track which k value this example corresponds to
            expanded_example['retrieval_k_value'] = k
            
            expanded_examples.append(expanded_example)
    
    # Convert back to HuggingFace dataset
    import datasets
    return datasets.Dataset.from_list(expanded_examples)


def load_dataset_with_local_support(**ds_kwargs):
    """
    Load dataset with support for local files/directories.
    
    This function extends the standard load_dataset to support:
    - Local HuggingFace dataset directories (load_from_disk)
    - Local JSONL files
    - Local JSON files
    - Regular HuggingFace Hub datasets (fallback)
    """
    try:
        # Check if 'path' points to a local file/directory
        if 'path' in ds_kwargs:
            path = ds_kwargs['path']
            if os.path.exists(path):
                # It's a local path, use our local loading function
                local_kwargs = {k: v for k, v in ds_kwargs.items() if k not in ['path']}
                logger.info(f"Loading local dataset from path: {path}")
                return load_local_dataset(path, **local_kwargs)
        
        # Check if 'data_files' points to local files
        if 'data_files' in ds_kwargs and ds_kwargs.get('path') != 'json':
            data_files = ds_kwargs['data_files']
            if isinstance(data_files, str) and os.path.exists(data_files):
                # It's a local file
                logger.info(f"Loading local dataset from data_files: {data_files}")
                return load_local_dataset(data_files, **{k: v for k, v in ds_kwargs.items() if k not in ['data_files']})
            elif isinstance(data_files, (list, dict)):
                # Check if all files in data_files are local
                files_to_check = []
                if isinstance(data_files, list):
                    files_to_check = data_files
                elif isinstance(data_files, dict):
                    files_to_check = [f for file_list in data_files.values() 
                                    for f in (file_list if isinstance(file_list, list) else [file_list])]
                
                if files_to_check and all(os.path.exists(f) for f in files_to_check):
                    # All files are local, use standard load_dataset with json loader
                    logger.info(f"Loading local files using standard loader: {files_to_check}")
                    return load_dataset(**ds_kwargs)
        
        # Fallback to standard load_dataset for HuggingFace Hub datasets
        logger.debug(f"Loading dataset from HuggingFace Hub with kwargs: {ds_kwargs}")
        return load_dataset(**ds_kwargs)
        
    except Exception as e:
        logger.error(f"Error loading dataset with kwargs {ds_kwargs}: {str(e)}")
        raise


class PerTaskEmbSFTDataset(Dataset):
    def __init__(self, tokenized_dataset: datasets.Dataset, task_embs: torch.Tensor, validation: bool):
        self.tokenized_dataset = tokenized_dataset
        self.task_embs = task_embs
        self.validation = validation

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        example = self.tokenized_dataset[idx]
        if self.task_embs is not None:
            emb_idx = idx % len(self.task_embs) if self.validation else random.randint(0, len(self.task_embs) - 1)
            task_emb = self.task_embs[emb_idx]
            example["task_emb"] = task_emb
        return example


class PerSampleEmbSFTDataset(Dataset):
    def __init__(
        self,
        tokenized_dataset: datasets.Dataset,
        task_embs,  # Can be torch.Tensor or dict for profile_text mode
        validation: bool,
        use_profile_text: bool = False,
        user_profile_format: str = "history",
        include_history_stat: bool = False,
    ):
        self.tokenized_dataset = tokenized_dataset
        self.task_embs = task_embs
        self.validation = validation
        self.use_profile_text = use_profile_text
        self.user_profile_format = user_profile_format
        self.include_history_stat = include_history_stat
        
        if use_profile_text and isinstance(task_embs, dict):
            # For profile_text mode with format_profile_text control
            self.profile_to_emb = task_embs['profile_to_emb']
            self.profile_lists = task_embs['profile_lists']
            
            # Verify correspondence between tokenized dataset and profile lists
            ds_name = getattr(tokenized_dataset, 'dataset_name', 'unknown_dataset')
            self.dataset_name = ds_name  # Store dataset name for use in _get_profile_key
            verify_profile_sample_correspondence(tokenized_dataset, self.profile_lists, ds_name)
            
            # Log tokenizer information if available
            if 'dataset_tokenizer_used' in task_embs and 'embedding_tokenizer_used' in task_embs:
                logger.info(f"Dataset loaded with tokenizer: {task_embs['dataset_tokenizer_used']}")
                logger.info(f"Embeddings created with tokenizer: {task_embs['embedding_tokenizer_used']}")
                
                # Warn if different tokenizers were used
                if task_embs['dataset_tokenizer_used'] != task_embs['embedding_tokenizer_used']:
                    logger.warning(f"Different tokenizers used for dataset loading and embedding generation. "
                                 f"This is expected and safe when using the safe profile extraction method.")
        elif not use_profile_text:
            # Traditional mode
            assert len(tokenized_dataset) == len(task_embs)
            self.dataset_name = getattr(tokenized_dataset, 'dataset_name', 'unknown_dataset')  # Store dataset name
    
    def _get_profile_key(self, profile_combination):
        """Get the formatted profile text based on user_profile_format for lookup."""
        from hyper_llm_modulator.utils.preprocessing import format_profile_text
        profile_text_str, profile_all_history_str, data_entry = profile_combination
        # Get profile_k from task_embs if available
        profile_k = self.task_embs.get('profile_k', 0) if isinstance(self.task_embs, dict) else 0
        return format_profile_text(profile_text_str, self.user_profile_format, profile_all_history_str, data_entry, profile_k, self.dataset_name, self.include_history_stat)

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        example = self.tokenized_dataset[idx]
        
        if self.use_profile_text and isinstance(self.task_embs, dict):
            # Use the profile combination (profile_text, profile_all_history) and let format_profile_text handle selection
            profile_combination = self.profile_lists[idx]
            
            # Create a key for this specific profile combination using user_profile_format
            profile_key = self._get_profile_key(profile_combination)
            task_emb = self.profile_to_emb[profile_key]
            example["formatted_profile_text"] = profile_key
        else:
            # Traditional mode or non-profile_text mode
            if self.task_embs is not None:
                task_emb = self.task_embs[idx]
            else:
                task_emb = None
        
        if task_emb is not None:
            example["task_emb"] = task_emb
        
        # Preserve additional fields if they exist (user_id, question_id)
        if self.use_profile_text:
            # These fields should be preserved from the original dataset if they exist
            for field in ["user_id", "question_id"]:
                if field in self.tokenized_dataset.column_names:
                    example[field] = self.tokenized_dataset[idx][field]
            
            # Store the original profile_text and profile_all_history for reference
            if "profile_text" in self.tokenized_dataset.column_names:
                example["profile_text"] = self.tokenized_dataset[idx]["profile_text"]
            if "profile_all_history" in self.tokenized_dataset.column_names:
                example["profile_all_history"] = self.tokenized_dataset[idx]["profile_all_history"]
        
        return example


class HierachicalBatchSampler(Sampler):
    # a sampler that first samples which dataset to sample from
    # then samples from that dataset
    # only works with ConcatDataset

    def __init__(self, concat_dataset: ConcatDataset, n_ds_per_batch: int, n_points_per_ds: int, dataset_weights=None):
        self.concat_dataset = concat_dataset
        self.n_ds_per_batch = n_ds_per_batch
        self.n_points_per_ds = n_points_per_ds
        self.cumulative_sizes = concat_dataset.cumulative_sizes
        self.n_datasets = len(self.cumulative_sizes)
        self.ds_sizes = [len(ds) for ds in concat_dataset.datasets]
        self.batch_size = n_ds_per_batch * n_points_per_ds

        # Ensure we emit at least one batch even if requested tasks-per-batch exceeds dataset count
        self.num_batches = max(1, self.n_datasets // self.n_ds_per_batch)
        if self.n_datasets < self.n_ds_per_batch:
            logger.warning(
                "Requested n_tasks_per_batch (%s) is larger than the number of available datasets (%s); "
                "sampling datasets with replacement to form batches.",
                self.n_ds_per_batch,
                self.n_datasets,
            )

        # Set up sampling weights
        if dataset_weights is not None:
            assert len(dataset_weights) == self.n_datasets, f"Number of weights ({len(dataset_weights)}) must match number of datasets ({self.n_datasets})"
            self.dataset_weights = torch.tensor(dataset_weights, dtype=torch.float32)
        else:
            # Default to uniform weights if none provided
            self.dataset_weights = torch.ones(self.n_datasets, dtype=torch.float32)
        
        # Normalize weights to sum to 1
        self.dataset_weights = self.dataset_weights / self.dataset_weights.sum()
        
        logger.info(f"HierarchicalBatchSampler initialized with {self.n_datasets} datasets")
        logger.info(f"Dataset sizes: {self.ds_sizes}")
        logger.info(f"Dataset sampling weights: {self.dataset_weights.tolist()}")

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # TODO: iterate over all samples in one epoch
        # Use weighted sampling instead of uniform permutation
        # Sample datasets with replacement according to weights, then shuffle
        samples_needed = max(self.n_datasets, self.n_ds_per_batch * self.num_batches)
        task_indices = torch.multinomial(
            self.dataset_weights,
            samples_needed,
            replacement=True
        )

        # Shuffle to randomize order within the epoch
        task_indices = task_indices[torch.randperm(len(task_indices))]

        for i in range(0, len(task_indices), self.n_ds_per_batch):
            batch_indices = []

            for j in range(i, i + self.n_ds_per_batch):
                if j >= len(task_indices):
                    break

                ds_idx = task_indices[j]
                ds_size = self.ds_sizes[ds_idx]
                local_indices = torch.randint(0, ds_size, (self.n_points_per_ds,))
                global_indices = local_indices + self.cumulative_sizes[ds_idx] - ds_size
                batch_indices.extend(global_indices.tolist())

            if len(batch_indices) == self.batch_size:
                yield batch_indices


def get_datasets(
    dataset_names,
    metadata,
    tokenizer,
    sft_mode,
    is_intx_model,
    inp_max_len,
    split_name=None,
):
    out = dict()
    dataset_info_dict = {k: metadata[k]["ds_kwargs"] for k in dataset_names}
    inp_tokenize_fn = get_inp_tokenize_fn(tokenizer, sft_mode, is_intx_model, inp_max_len)
    for i, (ds_name, ds_kwargs) in enumerate(dataset_info_dict.items()):
        logger.debug(f"ds_name: {ds_name}, ds_kwargs: {ds_kwargs}")
        # get hash for the dataset
        ds_repr = f"{ds_name}_{json.dumps(ds_kwargs)}_{tokenizer.name_or_path.strip('/')}_{sft_mode}_{is_intx_model}_{inp_max_len}"
        if split_name is not None:
            ds_repr += f"_{split_name}"

        ds_repr += f"_{json.dumps(metadata[ds_name])}"
        ds_hash = hashlib.sha256(ds_repr.encode("utf-8")).hexdigest()

        if glob(f"{TRANSFORMED_DS_DIR}/{ds_hash}/"):
            logger.debug(f"Loading preprocessed dataset: {ds_hash}")
            tokenized_dataset = datasets.load_from_disk(f"{TRANSFORMED_DS_DIR}/{ds_hash}")
            # Add dataset name as an attribute for later reference
            tokenized_dataset.dataset_name = ds_name
            # Update hash map for visibility
            _update_hash_map(
                TRANSFORMED_DS_DIR,
                ds_hash,
                ds_name,
                extra={
                    "context": "tokenized_dataset",
                    "tokenizer": tokenizer.name_or_path.strip('/'),
                    "sft_mode": sft_mode,
                    "is_intx_model": is_intx_model,
                    "inp_max_len": inp_max_len,
                    "ds_kwargs": ds_kwargs,
                    "split_name": split_name,
                },
            )
        else:
            formatted_dataset = load_and_format_dataset(
                metadata,
                tokenizer,
                sft_mode,
                is_intx_model,
                ds_name,
                ds_kwargs,
                split_name=split_name,
            )
            logger.debug(f"formatted example: {formatted_dataset[:5]}")
            
            # Preserve profile_text fields during tokenization
            profile_text_fields = ["user_id", "question_id", "profile_text", "profile_all_history"]
            columns_to_remove = [col for col in formatted_dataset.column_names 
                               if col not in profile_text_fields]
            
            tokenized_dataset = formatted_dataset.map(
                inp_tokenize_fn, batched=True, remove_columns=columns_to_remove
            )
            logger.debug(f"tokenized example: {tokenized_dataset[:5]}")
            tokenized_dataset.set_format("torch")

            logger.debug(f"Saving preprocessed dataset: {ds_hash}")
            tokenized_dataset.save_to_disk(f"{TRANSFORMED_DS_DIR}/{ds_hash}")
            _update_hash_map(
                TRANSFORMED_DS_DIR,
                ds_hash,
                ds_name,
                extra={
                    "context": "tokenized_dataset",
                    "tokenizer": tokenizer.name_or_path.strip('/'),
                    "sft_mode": sft_mode,
                    "is_intx_model": is_intx_model,
                    "inp_max_len": inp_max_len,
                    "ds_kwargs": ds_kwargs,
                    "split_name": split_name,
                },
            )

        # Add dataset name as an attribute for later reference
        tokenized_dataset.dataset_name = ds_name
        out[ds_name] = tokenized_dataset

    return out


def load_and_format_dataset(
    metadata,
    tokenizer,
    sft_mode,
    is_intx_model,
    ds_name,
    ds_kwargs,
    split_name=None,
):
    ds_repr = f"{ds_name}_{json.dumps(ds_kwargs)}_{tokenizer.name_or_path.strip('/')}_{sft_mode}_{is_intx_model}"
    if split_name is not None:
        ds_repr += f"_{split_name}"
    ds_repr += f"_{json.dumps(metadata[ds_name])}"
    ds_hash = hashlib.sha256(ds_repr.encode("utf-8")).hexdigest()
    if glob(f"{TRANSFORMED_DS_DIR}/{ds_hash}/"):
        logger.debug(f"Loading preprocessed dataset: {ds_hash}")
        formatted_dataset = datasets.load_from_disk(f"{TRANSFORMED_DS_DIR}/{ds_hash}")
        _update_hash_map(
            TRANSFORMED_DS_DIR,
            ds_hash,
            ds_name,
            extra={
                "context": "formatted_dataset",
                "tokenizer": tokenizer.name_or_path.strip('/'),
                "sft_mode": sft_mode,
                "is_intx_model": is_intx_model,
                "ds_kwargs": ds_kwargs,
                "split_name": split_name,
            },
        )
    else:
        dataset = load_dataset_with_local_support(**ds_kwargs)
        if split_name == "train":
            train_first_n = metadata[ds_name].get("train_first_n")
            if train_first_n is not None:
                if not isinstance(train_first_n, int):
                    raise ValueError(
                        f"Expected 'train_first_n' to be an int for task {ds_name}, got {type(train_first_n)}"
                    )
                if train_first_n > 0:
                    if hasattr(dataset, "select") and hasattr(dataset, "__len__"):
                        dataset_size = len(dataset)
                        capped_n = min(train_first_n, dataset_size)
                        if capped_n < dataset_size:
                            logger.info(
                                f"Limiting training dataset {ds_name} to the first {capped_n} examples "
                                f"(requested {train_first_n}, original size {dataset_size})"
                            )
                            dataset = dataset.select(range(capped_n))
                        else:
                            logger.info(
                                f"train_first_n for {ds_name} set to {train_first_n}, dataset size is {dataset_size}; no trimming applied"
                            )
                    else:
                        logger.warning(
                            f"Dataset object for {ds_name} does not support 'select'; unable to apply train_first_n"
                        )
                else:
                    logger.warning(
                        f"Ignoring non-positive train_first_n ({train_first_n}) for task {ds_name}"
                    )
        processed_dataset = dataset.map(get_preprocessing_fn(ds_name), batched=False)

        # Add retrieval_k expansion step for RAG tasks
        if ds_name.startswith('RAG_') and 'retrieval_k' in metadata[ds_name]:
            logger.info(f"Expanding dataset for retrieval_k values: {metadata[ds_name]['retrieval_k']}")
            processed_dataset = expand_dataset_for_retrieval_k(processed_dataset, metadata[ds_name])

        prompt_formatting_fn = get_prompt_formatting_fn(
            metadata[ds_name], sft_mode, tokenizer.apply_chat_template, is_intx_model
        )

        formatted_dataset = processed_dataset.map(prompt_formatting_fn, batched=True)
        formatted_dataset.save_to_disk(f"{TRANSFORMED_DS_DIR}/{ds_hash}")
        _update_hash_map(
            TRANSFORMED_DS_DIR,
            ds_hash,
            ds_name,
            extra={
                "context": "formatted_dataset",
                "tokenizer": tokenizer.name_or_path.strip('/'),
                "sft_mode": sft_mode,
                "is_intx_model": is_intx_model,
                "ds_kwargs": ds_kwargs,
                "split_name": split_name,
            },
        )
    return formatted_dataset


@torch.no_grad()
def get_task_embs(
    ds_descs,
    emb_model,
    emb_tokenizer,
    task_desc_format_fn,
    pooling_fn,
    device,
    use_api_embedding=False,
    api_embedding_kwargs=None,
):
    out = dict()
    for i, (ds_name, descs) in enumerate(ds_descs.items()):
        task_embs = None
        # pre-embed task descriptions when using per-task descriptions
        if emb_model is not None:
            # NOTE: assume that the number of task descs are small so we pad them here only once
            logger.debug(f"{ds_descs[ds_name]=}")
            task_embs = embed_texts(descs, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device,
                                    use_api_embedding=use_api_embedding, api_embedding_kwargs=api_embedding_kwargs)
        else:
            # one-hot task indicator
            task_embs = torch.eye(len(ds_descs), device=device)[i].unsqueeze(0)

        logger.debug(f"{task_embs=}")
        out[ds_name] = task_embs
    return out


def collator(inp_list, tokenizer):
    # input is a list of tokenized sequences
    padding_kwargs = dict(padding=True, pad_to_multiple_of=8, return_tensors="pt")
    labels = [x.pop("labels") for x in inp_list]
    task_embs = task_descs = None
    
    # Handle profile_text mode additional fields
    profile_text_fields = {}
    for field in ["user_id", "question_id", "profile_text", "profile_all_history", "sampled_profile_text", "formatted_profile_text"]:
        field_values = []
        for i, x in enumerate(inp_list):
            if field in x:
                field_values.append(x.pop(field))
            else:
                field_values.append(None)
        # Only store the field if at least one item has it
        if any(val is not None for val in field_values):
            profile_text_fields[field] = field_values
    
    # Handle task_emb field similarly
    task_embs_list = []
    for x in inp_list:
        if "task_emb" in x:
            task_embs_list.append(x.pop("task_emb"))
    if task_embs_list:
        task_embs = torch.stack(task_embs_list)
    
    padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

    # hacky explicit padding since the labels are not padded by default
    labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
    labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
    out = {**padded_seq, "labels": labels}
    if task_embs is not None:
        out["task_embs"] = task_embs
    
    # Add profile_text fields to output
    out.update(profile_text_fields)
    
    return out


def calculate_dataset_sampling_weights(ds_dict, sampling_strategy="sqrt_size", min_weight_ratio=0.1):
    """
    Calculate sampling weights for datasets to address imbalance.
    
    Args:
        ds_dict: Dictionary of dataset_name -> dataset
        sampling_strategy: Strategy for calculating weights
            - "uniform": Equal weights for all datasets (original behavior)
            - "size": Weights proportional to dataset size (larger datasets sampled more)
            - "sqrt_size": Weights proportional to square root of size (balanced approach)
            - "inv_size": Weights inversely proportional to size (smaller datasets sampled more)
        min_weight_ratio: Minimum weight ratio relative to max weight to prevent zero weights
    
    Returns:
        List of weights corresponding to datasets in ds_dict
    """
    dataset_sizes = [len(dataset) for dataset in ds_dict.values()]
    dataset_names = list(ds_dict.keys())
    
    if sampling_strategy == "uniform":
        weights = [1.0] * len(dataset_sizes)
    elif sampling_strategy == "size":
        weights = dataset_sizes
    elif sampling_strategy == "sqrt_size":
        import math
        weights = [math.sqrt(size) for size in dataset_sizes]
    elif sampling_strategy == "inv_size":
        max_size = max(dataset_sizes)
        weights = [max_size / size for size in dataset_sizes]
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    # Normalize weights and apply minimum weight ratio
    max_weight = max(weights)
    min_weight = max_weight * min_weight_ratio
    weights = [max(w, min_weight) for w in weights]
    
    # Log the weights for debugging
    logger.info(f"Dataset sampling strategy: {sampling_strategy}")
    for name, size, weight in zip(dataset_names, dataset_sizes, weights):
        logger.info(f"  {name}: size={size}, weight={weight:.4f}")

    return weights


def _downsample_tensor(tensor, indices):
    index_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index_tensor)


def _downsample_sequence(seq, indices):
    if isinstance(seq, tuple):
        return tuple(seq[i] for i in indices)
    return [seq[i] for i in indices]


def _downsample_profile_embeddings(emb_dict, indices):
    new_emb_dict = dict(emb_dict)
    if "profile_lists" in emb_dict:
        new_emb_dict["profile_lists"] = [emb_dict["profile_lists"][i] for i in indices]
    if "formatted_profiles" in emb_dict and len(emb_dict["formatted_profiles"]) == len(emb_dict["profile_lists"]):
        new_emb_dict["formatted_profiles"] = [emb_dict["formatted_profiles"][i] for i in indices]
    return new_emb_dict


def _subset_embeddings_structure(ds_embs, indices, original_length):
    """Apply the provided indices to the embedding structure when possible."""

    if ds_embs is None:
        return None

    effective_length = None
    if isinstance(ds_embs, dict) and "profile_lists" in ds_embs:
        try:
            effective_length = len(ds_embs["profile_lists"])
        except Exception:
            effective_length = None
    else:
        try:
            effective_length = len(ds_embs)
        except Exception:
            effective_length = None

    # Only subset structures that align with the dataset length to avoid misalignment.
    if effective_length is not None and effective_length != original_length:
        return ds_embs

    try:
        if isinstance(ds_embs, torch.Tensor):
            return _downsample_tensor(ds_embs, indices)
        if isinstance(ds_embs, datasets.Dataset):
            return ds_embs.select(indices)
        if isinstance(ds_embs, (list, tuple)):
            return _downsample_sequence(ds_embs, indices)
        if isinstance(ds_embs, dict) and "profile_lists" in ds_embs:
            return _downsample_profile_embeddings(ds_embs, indices)
        if hasattr(ds_embs, "select") and effective_length is not None and effective_length == original_length:
            return ds_embs.select(indices)
    except Exception as exc:
        logger.warning(f"Failed to subset embeddings for training cluster filter: {exc}")

    return ds_embs


def _normalize_ds_name_for_embeddings(ds_name: str, metadata_entry: dict) -> str:
    """Derive the embedding file stem from dataset metadata."""

    candidate = metadata_entry.get("ds_kwargs", {}).get("name") or ds_name
    candidate = candidate.lower()

    # Remove common prefixes used for alternate dataset variants
    for prefix in ("all_history_", "profile_", "rag_"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]

    # Iteratively strip known suffixes until none remain
    suffixes = [
        "_train",
        "_random_train",
        "_random_test",
        "_ood_train",
        "_ood_test",
        "_eval",
        "_validation",
        "_val",
        "_dev",
    ]
    updated = True
    while updated:
        updated = False
        for suffix in suffixes:
            if candidate.endswith(suffix):
                candidate = candidate[: -len(suffix)]
                updated = True
                break

    return candidate


def _filter_dataset_by_clusters(
    ds_name,
    dataset,
    dataset_embs,
    metadata_entry,
    embeddings_dir,
    total_clusters,
    clusters_in_train,
    selection_strategy,
    base_seed,
    user_cap,
):
    """Filter a single dataset according to per-user clustering configuration."""

    if not isinstance(dataset, datasets.Dataset):
        return dataset, dataset_embs

    if "user_id" not in dataset.column_names or len(dataset) == 0:
        return dataset, dataset_embs

    user_ids = dataset["user_id"]
    user_to_indices = defaultdict(list)
    for idx, uid in enumerate(user_ids):
        user_to_indices[uid].append(idx)

    unique_user_ids = list(user_to_indices.keys())
    if len(unique_user_ids) == 0:
        return dataset, dataset_embs

    embedding_stem = _normalize_ds_name_for_embeddings(ds_name, metadata_entry)
    embedding_file = Path(embeddings_dir) / f"{embedding_stem}_user_embeddings.npz"
    if not embedding_file.exists():
        logger.warning(
            f"User cluster filtering skipped for {ds_name}: embedding file not found at {embedding_file}"
        )
        return dataset, dataset_embs

    try:
        emb_data = np.load(embedding_file, allow_pickle=True)
        embedding_user_ids = [str(uid) for uid in emb_data["user_ids"]]
        embeddings = emb_data["embeddings"]
    except Exception as exc:
        logger.warning(
            f"User cluster filtering skipped for {ds_name}: failed to load embeddings from {embedding_file}: {exc}"
        )
        return dataset, dataset_embs

    id_to_idx = {uid: idx for idx, uid in enumerate(embedding_user_ids)}
    clustered_user_ids = [uid for uid in unique_user_ids if uid in id_to_idx]
    missing_user_ids = [uid for uid in unique_user_ids if uid not in id_to_idx]

    if len(clustered_user_ids) == 0:
        logger.warning(
            f"User cluster filtering skipped for {ds_name}: no matching user embeddings were found"
        )
        return dataset, dataset_embs

    cluster_total = min(total_clusters, len(clustered_user_ids))
    if cluster_total <= 0:
        return dataset, dataset_embs

    cluster_train = min(clusters_in_train, cluster_total)
    if cluster_train <= 0:
        return dataset, dataset_embs

    # Prepare user embedding matrix in dataset order for clustering
    embed_indices = [id_to_idx[uid] for uid in clustered_user_ids]
    user_vectors = embeddings[embed_indices]

    modulus = 2 ** 31 - 1
    cluster_seed = (abs(hash(ds_name)) + base_seed) % modulus
    try:
        cluster_model = MiniBatchKMeans(
            n_clusters=cluster_total,
            random_state=cluster_seed,
            batch_size=min(2048, len(clustered_user_ids)),
            n_init="auto",
        )
    except TypeError:
        # Older sklearn versions do not support n_init="auto"
        cluster_model = MiniBatchKMeans(
            n_clusters=cluster_total,
            random_state=cluster_seed,
            batch_size=min(2048, len(clustered_user_ids)),
        )

    try:
        cluster_model.fit(user_vectors)
    except Exception as exc:
        logger.warning(
            f"User cluster filtering skipped for {ds_name}: clustering failed with error {exc}"
        )
        return dataset, dataset_embs

    labels = cluster_model.labels_.astype(int)
    cluster_to_users = defaultdict(list)
    for uid, label in zip(clustered_user_ids, labels):
        cluster_to_users[label].append(uid)

    if not cluster_to_users:
        return dataset, dataset_embs

    cluster_items = list(cluster_to_users.items())
    rng = random.Random(cluster_seed)
    if selection_strategy == "random":
        rng.shuffle(cluster_items)
    elif selection_strategy == "smallest":
        cluster_items.sort(key=lambda item: (len(item[1]), item[0]))
    else:
        cluster_items.sort(key=lambda item: (-len(item[1]), item[0]))

    selected_cluster_items = cluster_items[:cluster_train]
    if not selected_cluster_items:
        return dataset, dataset_embs

    # Build ordered user lists so sampling choices are deterministic under the RNG.
    missing_unique = []
    seen_missing = set()
    for uid in missing_user_ids:
        if uid not in seen_missing:
            seen_missing.add(uid)
            missing_unique.append(uid)

    cluster_users_ordered = []
    seen_cluster_users = set()
    for _, users in selected_cluster_items:
        for uid in users:
            if uid not in seen_cluster_users:
                seen_cluster_users.add(uid)
                cluster_users_ordered.append(uid)

    selected_user_set = set(seen_missing)
    selected_user_set.update(seen_cluster_users)

    if user_cap is not None and user_cap > 0 and len(selected_user_set) < user_cap:
        logger.warning(
            f"User cluster filtering for {ds_name}: only {len(selected_user_set)} unique users available "
            f"after clustering (requested cap={user_cap})."
        )

    if user_cap is not None and user_cap > 0 and len(selected_user_set) > user_cap:
        rng_missing = missing_unique[:]
        rng.shuffle(rng_missing)
        retained_users = []

        if rng_missing:
            retained_missing = rng_missing[: min(len(rng_missing), user_cap)]
            retained_users.extend(retained_missing)

        remaining_slots = user_cap - len(retained_users)
        if remaining_slots > 0:
            rng_cluster_users = cluster_users_ordered[:]
            rng.shuffle(rng_cluster_users)
            retained_users.extend(rng_cluster_users[: remaining_slots])

        if len(retained_users) < user_cap and len(selected_user_set) <= user_cap:
            retained_users = list(selected_user_set)

        selected_user_set = set(retained_users)
        logger.info(
            f"User cluster filtering for {ds_name}: downsampled unique users to {len(selected_user_set)} "
            f"(cap={user_cap})."
        )

    if user_cap is None and len(selected_user_set) == len(unique_user_ids):
        # All users retained – no changes necessary
        return dataset, dataset_embs

    candidate_indices = []
    for idx, uid in enumerate(user_ids):
        if uid in selected_user_set:
            candidate_indices.append(idx)

    if not candidate_indices:
        logger.warning(
            f"User cluster filtering skipped for {ds_name}: selected clusters produced zero training samples"
        )
        return dataset, dataset_embs

    original_length = len(dataset)
    if len(candidate_indices) < original_length:
        deficit = original_length - len(candidate_indices)
        if deficit > 0 and len(candidate_indices) > 0:
            replicated_indices = rng.choices(candidate_indices, k=deficit)
            candidate_indices.extend(replicated_indices)
            logger.info(
                f"User cluster filtering for {ds_name}: retained {len(selected_user_set)} unique users across "
                f"{cluster_train}/{cluster_total} clusters and duplicated {deficit} samples to preserve "
                f"the original dataset size ({original_length})."
            )
    elif len(candidate_indices) > original_length:
        rng.shuffle(candidate_indices)
        candidate_indices = candidate_indices[:original_length]

    candidate_indices.sort()

    filtered_dataset = dataset.select(candidate_indices)
    filtered_dataset.dataset_name = getattr(dataset, "dataset_name", ds_name)
    filtered_embs = _subset_embeddings_structure(dataset_embs, candidate_indices, original_length)

    logger.info(
        f"User cluster filtering applied to {ds_name}: {cluster_train}/{cluster_total} clusters used, "
        f"{len(set(user_ids[idx] for idx in candidate_indices))} unique users retained out of "
        f"{len(unique_user_ids)}."
    )

    return filtered_dataset, filtered_embs


def filter_training_datasets_by_user_clusters(args, ds_dict, ds_embs_dict, metadata):
    """Apply user clustering-based filtering to the training datasets when requested."""

    total_clusters = getattr(args, "train_user_total_clusters", None)
    clusters_in_train = getattr(args, "train_user_clusters_in_train", None)

    if not total_clusters or not clusters_in_train:
        return ds_dict, ds_embs_dict

    if total_clusters <= 0 or clusters_in_train <= 0:
        return ds_dict, ds_embs_dict

    embeddings_dir = getattr(args, "train_user_cluster_embeddings_dir", DEFAULT_TRAIN_USER_EMBEDDINGS_DIR)
    selection_strategy = getattr(args, "train_user_cluster_selection_strategy", "largest")
    user_cap = getattr(args, "train_user_cluster_user_cap", None)
    cluster_seed = getattr(args, "train_user_cluster_seed", None)
    if cluster_seed is None:
        cluster_seed = getattr(args, "seed", 0) or 0

    updated_ds_dict = {}
    updated_embs_dict = {} if ds_embs_dict is not None else None

    for ds_name, dataset in ds_dict.items():
        ds_metadata = metadata.get(ds_name, {}) if metadata else {}
        dataset_embs = ds_embs_dict.get(ds_name) if ds_embs_dict else None

        filtered_dataset, filtered_embs = _filter_dataset_by_clusters(
            ds_name,
            dataset,
            dataset_embs,
            ds_metadata,
            embeddings_dir,
            total_clusters,
            clusters_in_train,
            selection_strategy,
            cluster_seed,
            user_cap,
        )

        updated_ds_dict[ds_name] = filtered_dataset
        if updated_embs_dict is not None:
            updated_embs_dict[ds_name] = filtered_embs

    return updated_ds_dict, updated_embs_dict


def downsample_training_datasets(ds_dict, ds_embs_dict, proportion, seed):
    """Randomly downsample the *combined* training datasets to the requested proportion.

    The target number of samples is computed across all training datasets. Individual datasets
    receive a share of the target count proportional to their size, and any remainder is
    distributed randomly for reproducibility. Embedding tensors/datasets are downsampled using
    the same indices to preserve alignment.
    """

    if not (0 < proportion <= 1):
        raise ValueError("train_data_proportion must be in the interval (0, 1].")

    total_size = sum(len(dataset) for dataset in ds_dict.values())
    if total_size == 0:
        logger.warning("All training datasets are empty; skipping downsampling")
        return ds_dict, ds_embs_dict

    if proportion >= 1:
        return ds_dict, ds_embs_dict

    dataset_sizes = {ds_name: len(dataset) for ds_name, dataset in ds_dict.items()}
    min_required = sum(1 for size in dataset_sizes.values() if size > 0)

    target_total = max(1, int(total_size * proportion))
    target_total = max(target_total, min_required)
    if target_total >= total_size:
        return ds_dict, ds_embs_dict

    logger.info(
        f"Downsampling training datasets to {target_total}/{total_size} samples ({proportion:.2%})"
    )

    desired_counts = {}
    samples_to_keep = {}
    fractional_remainders = {}

    for ds_name, size in dataset_sizes.items():
        desired = size * proportion
        desired_counts[ds_name] = desired
        base = min(size, int(desired))
        keep = base

        if size > 0 and keep == 0:
            keep = 1

        samples_to_keep[ds_name] = keep

        fractional = max(desired - base, 0.0)
        if keep >= size or (base == 0 and keep == 1):
            fractional = 0.0
        fractional_remainders[ds_name] = fractional

    current_total = sum(samples_to_keep.values())

    if current_total < target_total:
        deficit = target_total - current_total
        allocation_rng = random.Random(seed)
        candidates = [
            name for name, size in dataset_sizes.items() if samples_to_keep[name] < size
        ]
        weights = [fractional_remainders[name] for name in candidates]

        if sum(weights) <= 0:
            weights = None

        while deficit > 0 and candidates:
            if weights is None:
                choice = allocation_rng.choice(candidates)
            else:
                choice = allocation_rng.choices(candidates, weights=weights, k=1)[0]

            samples_to_keep[choice] += 1
            deficit -= 1

            idx = candidates.index(choice)
            if samples_to_keep[choice] >= dataset_sizes[choice]:
                candidates.pop(idx)
                if weights is not None:
                    weights.pop(idx)
            elif weights is not None:
                weights[idx] = max(weights[idx] - 1, 0.0)
                if sum(weights) <= 0:
                    weights = None

        if deficit > 0:
            logger.warning(
                "Requested train_data_proportion exhausted available samples while adding data; using full datasets."
            )

    elif current_total > target_total:
        surplus = current_total - target_total
        removal_seed = seed + len(dataset_sizes) if seed is not None else None
        removal_rng = random.Random(removal_seed)
        removable = [name for name, keep in samples_to_keep.items() if keep > 1]
        removal_weights = [
            max(samples_to_keep[name] - desired_counts[name], 0.0) for name in removable
        ]

        if sum(removal_weights) <= 0:
            removal_weights = None

        while surplus > 0 and removable:
            if removal_weights is None:
                choice = removal_rng.choice(removable)
            else:
                choice = removal_rng.choices(removable, weights=removal_weights, k=1)[0]

            samples_to_keep[choice] -= 1
            surplus -= 1

            idx = removable.index(choice)
            if samples_to_keep[choice] <= 1:
                removable.pop(idx)
                if removal_weights is not None:
                    removal_weights.pop(idx)
            elif removal_weights is not None:
                removal_weights[idx] = max(
                    samples_to_keep[choice] - desired_counts[choice], 0.0
                )
                if sum(removal_weights) <= 0:
                    removal_weights = None

        if surplus > 0:
            logger.warning(
                "Unable to honor train_data_proportion exactly because each dataset must retain at least one sample."
            )

    updated_ds_dict = {}
    updated_embs_dict = None if ds_embs_dict is None else {}
    total_selected = 0

    for idx, (ds_name, dataset) in enumerate(ds_dict.items()):
        keep = samples_to_keep.get(ds_name, 0)
        total = dataset_sizes[ds_name]

        if keep <= 0:
            continue

        if keep >= total:
            updated_ds_dict[ds_name] = dataset
            if updated_embs_dict is not None:
                updated_embs_dict[ds_name] = ds_embs_dict.get(ds_name)
            logger.info(f"Downsampled {ds_name}: {total} -> {keep} samples (no reduction)")
            total_selected += total
            continue

        dataset_seed = seed + idx if seed is not None else None
        rng = random.Random(dataset_seed)
        indices = rng.sample(range(total), keep)
        indices.sort()

        subset = dataset.select(indices)
        subset.dataset_name = getattr(dataset, "dataset_name", ds_name)
        updated_ds_dict[ds_name] = subset

        if updated_embs_dict is not None:
            ds_embs = ds_embs_dict.get(ds_name)
            new_embs = ds_embs
            try:
                if ds_embs is None:
                    new_embs = None
                elif isinstance(ds_embs, torch.Tensor) and ds_embs.size(0) == total:
                    new_embs = _downsample_tensor(ds_embs, indices)
                elif isinstance(ds_embs, datasets.Dataset) and len(ds_embs) == total:
                    new_embs = ds_embs.select(indices)
                elif isinstance(ds_embs, (list, tuple)) and len(ds_embs) == total:
                    new_embs = _downsample_sequence(ds_embs, indices)
                elif isinstance(ds_embs, dict) and "profile_lists" in ds_embs:
                    new_embs = _downsample_profile_embeddings(ds_embs, indices)
                elif hasattr(ds_embs, "select") and len(ds_embs) == total:
                    new_embs = ds_embs.select(indices)
                else:
                    logger.debug(
                        f"Skipping embedding downsampling for {ds_name} due to incompatible shape or type"
                    )
            except Exception as exc:
                logger.warning(f"Failed to downsample embeddings for {ds_name}: {exc}")
                new_embs = ds_embs

            updated_embs_dict[ds_name] = new_embs

        logger.info(
            f"Downsampled {ds_name}: {total} -> {keep} samples (seed={dataset_seed})"
        )
        total_selected += keep

    logger.info(
        f"Total training samples after downsampling: {total_selected}/{total_size}"
    )

    return updated_ds_dict, updated_embs_dict


@torch.no_grad()
def get_dataloader(
    ds_dict,
    task_embs_dict,
    tokenizer,
    use_per_task_emb,
    use_inp_as_desc,
    use_per_sample_desc,
    n_tasks_per_batch,
    n_points_per_task,
    use_hierarchical_sampler,
    batch_size,  # only needed for random sampler
    validation,
    use_profile_text=False,  # New parameter for profile_text mode
    dataset_sampling_strategy="sqrt_size",  # New parameter for dataset sampling strategy
    include_history_stat=False,  # New parameter for including history statistics
):
    if task_embs_dict is not None:
        assert len(ds_dict) == len(task_embs_dict)

    ds_list = []
    for ds_name in ds_dict:
        if use_profile_text:
            # When using profile_text, always use PerSampleEmbSFTDataset for deterministic correspondence
            user_profile_format = task_embs_dict[ds_name].get('user_profile_format', 'history') if isinstance(task_embs_dict[ds_name], dict) else 'history'
            ds_list.append(PerSampleEmbSFTDataset(
                ds_dict[ds_name], 
                task_embs_dict[ds_name], 
                validation, 
                use_profile_text=True,
                user_profile_format=user_profile_format,
                include_history_stat=include_history_stat
            ))
        elif use_per_task_emb:
            ds_list.append(PerTaskEmbSFTDataset(ds_dict[ds_name], task_embs_dict[ds_name], validation))
        elif use_inp_as_desc or use_per_sample_desc:
            ds_list.append(PerSampleEmbSFTDataset(ds_dict[ds_name], task_embs_dict[ds_name], validation, use_profile_text=False, user_profile_format="history", include_history_stat=include_history_stat))
        else:
            # no-op
            ds_list.append(ds_dict[ds_name])

    dataset = torch.utils.data.ConcatDataset(ds_list)
    if use_hierarchical_sampler:
        # Calculate dataset sampling weights to address imbalance
        dataset_weights = calculate_dataset_sampling_weights(ds_dict, dataset_sampling_strategy)
        sampler = HierachicalBatchSampler(dataset, n_tasks_per_batch, n_points_per_task, dataset_weights)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
        sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=partial(collator, tokenizer=tokenizer))


def create_dataloaders(
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
):
    # Prepare API embedding configuration if enabled
    use_api_embedding = getattr(args, 'use_api_embedding', False)
    api_embedding_kwargs = None
    
    if use_api_embedding:
        from hyper_llm_modulator.utils.api_embedding import create_api_embedding_kwargs
        api_embedding_kwargs = create_api_embedding_kwargs(args)
        logger.info("API embedding enabled")

    _get_datasets = partial(
        get_datasets,
        tokenizer=tokenizer,
        sft_mode=args.sft_mode,
        is_intx_model=is_intx_model,
        inp_max_len=args.inp_max_len,
    )
    
    # Note: use_profile_text will be determined per split in the loop below
    # to ensure eval datasets can use profile_text even if training datasets don't
    _get_dataloader_base = partial(
        get_dataloader,
        tokenizer=tokenizer,
        use_per_task_emb=args.use_per_task_emb,
        use_inp_as_desc=args.use_inp_as_desc,
        use_per_sample_desc=args.use_per_sample_desc,
        n_tasks_per_batch=args.n_tasks_per_batch,
        n_points_per_task=args.n_points_per_task,
        include_history_stat=getattr(args, 'include_history_stat', False),
    )

    if emb_model is not None:
        emb_model = emb_model.eval()

    val_ds_names = []
    benchmark_val_ds_names = []
    unseen_val_ds_names = []

    for ds_name in args.train_ds_names:
        # by default, we load max 10,000 samples for each task (see tasks/lol_*/metadata.yaml)
        # which, as far as i can tell, does not actually cap the number of samples
        if ds_name in args.eval_ds_info:
            # make a validation split for tasks that are in both training and validation
            train_metadata[ds_name]["ds_kwargs"]["split"] = "train[:90%]"
            if ds_name == "longreward":
                train_metadata[ds_name]["ds_kwargs"]["split"] = "sft[:90%]"

    val_ds_names = [name for name in args.eval_ds_info if name in args.train_ds_names]
    for ds_name in val_ds_names:
        val_metadata[ds_name]["ds_kwargs"]["split"] = "train[90%:]"
        if ds_name == "longreward":
            val_metadata[ds_name]["ds_kwargs"]["split"] = "sft[90%:]"

    if use_hypernet or "mt" in args.exp_setup:
        # meta-validation datasets
        unseen_val_ds_names = [
            name for name in args.eval_ds_info if name not in args.train_ds_names
        ]
        benchmark_val_ds_names = [t for t in BENCHMARK_TASK_INFO if t in val_metadata]
        for ds_name in unseen_val_ds_names:
            val_metadata[ds_name]["ds_kwargs"]["split"] = "valid[:500]"
        for ds_name in benchmark_val_ds_names:
            val_metadata[ds_name]["ds_kwargs"].update(BENCHMARK_TASK_INFO[ds_name])

    out = {"train": None, "val/seen": None, "val/unseen": None, "val/benchmark": None}
    ds_names_list = [args.train_ds_names, val_ds_names, unseen_val_ds_names, benchmark_val_ds_names]

    logging.info(f"{args.use_per_task_emb=}\n{args.use_inp_as_desc=}\n{args.use_per_sample_desc=}")
    
    for split_name, ds_names in zip(out, ds_names_list):
        logger.info(f"{split_name=}, {ds_names=}")
        if len(ds_names) == 0:
            continue
        metadata = train_metadata if split_name == "train" else val_metadata
        
        # Check if this specific split should use profile_text mode
        # This ensures eval datasets can use profile_text even if training datasets don't
        use_profile_text_for_split = check_datasets_for_profile_text(ds_names, metadata)
        
        if use_profile_text_for_split:
            logger.info(f"PROFILE_TEXT MODE ENABLED for {split_name}: Using profile_text lists from datasets with random sampling as hypernetwork input instead of task descriptions")
        else:
            logger.info(f"Using traditional task descriptions from metadata for hypernetwork input for {split_name}")
        
        ds_dict = _get_datasets(ds_names, metadata, split_name=split_name)
        ds_embs_dict = get_embs_dict(
            args,
            emb_model,
            emb_tokenizer,
            task_desc_format_fn,
            pooling_fn,
            ds_names,
            metadata,
            device,
            main_tokenizer=tokenizer,  # Pass main tokenizer for consistency
            use_api_embedding=use_api_embedding,
            api_embedding_kwargs=api_embedding_kwargs,
        )

        if split_name == "train":
            ds_dict, ds_embs_dict = filter_training_datasets_by_user_clusters(args, ds_dict, ds_embs_dict, metadata)
            train_prop = getattr(args, "train_data_proportion", 1.0)
            ds_dict, ds_embs_dict = downsample_training_datasets(
                ds_dict,
                ds_embs_dict,
                train_prop,
                seed=getattr(args, "seed", None),
            )

        train_kwargs = dict(
            use_hierarchical_sampler=args.use_hierarchical_sampler,
            batch_size=args.batch_size,
            validation=False,
            use_profile_text=use_profile_text_for_split,
            dataset_sampling_strategy=getattr(args, 'dataset_sampling_strategy', 'sqrt_size'),
        )
        val_kwargs = dict(
            use_hierarchical_sampler=False, 
            batch_size=args.val_batch_size, 
            validation=True,
            use_profile_text=use_profile_text_for_split,
            dataset_sampling_strategy=getattr(args, 'dataset_sampling_strategy', 'sqrt_size'),
        )
        kwargs = train_kwargs if split_name == "train" else val_kwargs

        out[split_name] = _get_dataloader_base(ds_dict, ds_embs_dict, **kwargs)

    return out


def check_datasets_for_profile_text(ds_names, metadata):
    """Check if any of the datasets have profile_text field in their data."""
    for ds_name in ds_names:
        try:
            # Load a small sample to check the structure
            ds_kwargs = metadata[ds_name]["ds_kwargs"]
            sample_kwargs = ds_kwargs.copy()
            # Just check the first few rows to see if profile_text exists
            if "split" in sample_kwargs:
                sample_kwargs["split"] = sample_kwargs["split"].split(":")[0] + "[:5]"
            else:
                sample_kwargs["split"] = "train[:5]"
            
            sample_dataset = load_dataset_with_local_support(**sample_kwargs)
            if isinstance(sample_dataset, dict):
                # Handle case where load_dataset returns a DatasetDict
                sample_dataset = list(sample_dataset.values())[0]
            
            # Check if profile_text exists in the dataset columns
            if "profile_text" in sample_dataset.column_names:
                return True
        except Exception as e:
            logger.debug(f"Error checking dataset {ds_name} for profile_text: {e}")
            continue
    return False


def verify_profile_sample_correspondence(tokenized_dataset, profile_combinations, ds_name):
    """
    Verify that profile combinations correspond to the correct samples by checking
    user_id and question_id if available.
    """
    # Handle unknown dataset case more gracefully
    if ds_name == 'unknown_dataset':
        logger.warning("Dataset name is unknown - this may happen with cached datasets. "
                      "Proceeding with verification using available fields.")
    
    if len(tokenized_dataset) != len(profile_combinations):
        raise ValueError(f"Length mismatch in {ds_name}: tokenized_dataset={len(tokenized_dataset)}, profile_combinations={len(profile_combinations)}")
    
    # If user_id and question_id are available, verify correspondence
    if "user_id" in tokenized_dataset.column_names and "question_id" in tokenized_dataset.column_names:
        logger.info(f"Verifying profile correspondence for {ds_name} using user_id and question_id...")
        
        # Sample a few indices to verify correspondence
        check_indices = [0, len(tokenized_dataset)//4, len(tokenized_dataset)//2, len(tokenized_dataset)-1]
        check_indices = [i for i in check_indices if i < len(tokenized_dataset)]
        
        for idx in check_indices:
            tokenized_sample = tokenized_dataset[idx]
            
            # Check profile_text correspondence
            if "profile_text" in tokenized_dataset.column_names:
                expected_profile_text = str(tokenized_sample["profile_text"]) if tokenized_sample["profile_text"] is not None else ""
                actual_profile_text = profile_combinations[idx][0]  # First element of tuple
                
                if expected_profile_text != actual_profile_text:
                    logger.error(f"Profile text mismatch at index {idx} in {ds_name}")
                    logger.error(f"Expected: {expected_profile_text}")
                    logger.error(f"Actual: {actual_profile_text}")
                    raise ValueError(f"Profile text mismatch detected at index {idx} in {ds_name}")
            
            # Check profile_all_history correspondence if available
            if "profile_all_history" in tokenized_dataset.column_names:
                expected_profile_all_history = str(tokenized_sample["profile_all_history"]) if tokenized_sample["profile_all_history"] is not None else ""
                actual_profile_all_history = profile_combinations[idx][1]  # Second element of tuple
                
                if expected_profile_all_history != actual_profile_all_history:
                    logger.error(f"Profile all history mismatch at index {idx} in {ds_name}")
                    logger.error(f"Expected: {expected_profile_all_history}")
                    logger.error(f"Actual: {actual_profile_all_history}")
                    raise ValueError(f"Profile all history mismatch detected at index {idx} in {ds_name}")
            
            # Check profile retrieval fields if they exist (third element of tuple contains data_entry)
            if len(profile_combinations[idx]) >= 3:
                data_entry = profile_combinations[idx][2]  # Third element contains data_entry dict
                for retrieval_field, expected_value in data_entry.items():
                    if retrieval_field in tokenized_dataset.column_names:
                        actual_value = str(tokenized_sample[retrieval_field]) if tokenized_sample[retrieval_field] is not None else ""
                        expected_value_str = str(expected_value) if expected_value is not None else ""
                        
                        if actual_value != expected_value_str:
                            logger.error(f"Profile {retrieval_field} mismatch at index {idx} in {ds_name}")
                            logger.error(f"Expected: {expected_value_str}")
                            logger.error(f"Actual: {actual_value}")
                            raise ValueError(f"Profile {retrieval_field} mismatch detected at index {idx} in {ds_name}")
        
        logger.info(f"Profile correspondence verification passed for {ds_name}")
    else:
        if ds_name != 'unknown_dataset':
            logger.warning(f"Cannot verify profile correspondence for {ds_name} - user_id/question_id not available")
        else:
            logger.debug("Cannot verify profile correspondence - user_id/question_id not available and dataset name unknown")


def get_profile_text_embs_dict_safe(args, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, metadata, device, main_tokenizer=None, use_api_embedding=False, api_embedding_kwargs=None, include_history_stat=False):
    """
    Safe version that ensures profile text extraction uses consistent tokenization.
    
    This function addresses the potential issue where different tokenizers could lead to 
    different dataset orderings by ensuring we use the same tokenizer for both profile
    extraction and training dataset creation when possible.
    
    Args:
        main_tokenizer: The main tokenizer used for training dataset creation.
                       If provided and different from emb_tokenizer, we'll use it 
                       for dataset loading to ensure consistency.
        include_history_stat: Whether to include history statistics in the embedding input.
    """
    # Handle case when emb_model is None (e.g., in mt-lora mode)
    if emb_model is None:
        logger.info("emb_model is None, returning None embeddings for all datasets (e.g., mt-lora mode)")
        return {ds_name: None for ds_name in ds_names}
    
    ds_embs_dict = {}
    
    # Determine which tokenizer to use for dataset loading
    dataset_tokenizer = main_tokenizer if main_tokenizer is not None else emb_tokenizer
    
    if main_tokenizer is not None and main_tokenizer != emb_tokenizer:
        emb_tokenizer_name = emb_tokenizer.name_or_path if emb_tokenizer is not None else "None"
        logger.warning(f"Using main tokenizer ({main_tokenizer.name_or_path}) for dataset loading "
                      f"instead of embedding tokenizer ({emb_tokenizer_name}) to ensure "
                      f"consistent sample ordering between profile extraction and training.")
    
    for ds_name in ds_names:
        # Use the same tokenizer as will be used for training dataset creation
        formatted_dataset = load_and_format_dataset(
            metadata,
            dataset_tokenizer,  # Use consistent tokenizer
            args.sft_mode,
            is_intx_model=dataset_tokenizer.chat_template is not None,
            ds_name=ds_name,
            ds_kwargs=metadata[ds_name]["ds_kwargs"],
        )
        
        # Extract profile_text for each sample
        if "profile_text" in formatted_dataset.column_names:
            profile_texts = formatted_dataset["profile_text"]  # Now strings, not lists
            
            # Define datasets that only have profile_text (no profile_all_history or profile_retrieval_k fields)
            profile_text_only_datasets = ["personalreddit", "opinionqa", "prism", "aloe", "ec"]
            is_profile_text_only = any(dataset_name in ds_name.lower() for dataset_name in profile_text_only_datasets)
            
            # Check if profile_all_history column exists
            if "profile_all_history" in formatted_dataset.column_names:
                profile_all_histories = formatted_dataset["profile_all_history"]
            else:
                if is_profile_text_only:
                    # For datasets that only have profile_text, use profile_text as the history
                    logger.info(f"Dataset {ds_name} only has profile_text field. Using profile_text for embedding generation.")
                    profile_all_histories = profile_texts  # Use profile_text as the history
                else:
                    raise ValueError(f"profile_all_history column not found in dataset {ds_name}. Please check the dataset.")

            # Get profile_k parameter from args
            profile_k = getattr(args, 'profile_k', 0)
            
            # Get profile retrieval fields if profile_k > 0
            profile_retrieval_fields = {}
            if profile_k > 0:
                profile_retrieval_key = f"profile_retrieval_k{profile_k}"
                if profile_retrieval_key in formatted_dataset.column_names:
                    profile_retrieval_fields[profile_retrieval_key] = formatted_dataset[profile_retrieval_key]
                else:
                    # If the specific retrieval field is missing, fall back to profile_text
                    # instead of profile_all_history per requirement.
                    logger.warning(
                        f"profile_k={profile_k} specified but {profile_retrieval_key} field not found in dataset. "
                        f"Using profile_text as fallback."
                    )
                    profile_retrieval_fields[profile_retrieval_key] = profile_texts

            # Optionally gather per-sample history statistics (for certain LaMP tasks)
            history_stats = None
            if include_history_stat and "history_stat" in formatted_dataset.column_names:
                # Limit to movie/news_cat tasks per config intent
                if ("lamp_movie" in ds_name) or ("lamp_news_cat" in ds_name) or ("movie" in ds_name and "lamp" in ds_name.lower()) or ("news_cat" in ds_name):
                    history_stats = formatted_dataset["history_stat"]
            
            # Get all unique profile combinations to avoid duplicate embeddings
            unique_profile_combinations = []
            seen_combinations = set()
            
            for i in range(len(profile_texts)):
                profile_text = profile_texts[i]
                profile_all_history = profile_all_histories[i]
                
                # Ensure both are strings
                profile_text_str = str(profile_text) if profile_text is not None else ""
                profile_all_history_str = str(profile_all_history) if profile_all_history is not None else ""
                
                # Create data entry for profile_k lookup
                data_entry = {}
                if profile_k > 0 and profile_retrieval_fields:
                    profile_retrieval_key = f"profile_retrieval_k{profile_k}"
                    if profile_retrieval_key in profile_retrieval_fields:
                        data_entry[profile_retrieval_key] = profile_retrieval_fields[profile_retrieval_key][i]

                # Attach history statistics if available
                if history_stats is not None:
                    try:
                        data_entry["history_stat"] = history_stats[i]
                    except Exception:
                        # Best-effort; skip if indexing mismatch
                        pass
                
                # Create a combination key that includes all relevant data for deduplication
                combination_key = (profile_text_str, profile_all_history_str, tuple(sorted(data_entry.items())))
                if combination_key not in seen_combinations:
                    seen_combinations.add(combination_key)
                    unique_profile_combinations.append((profile_text_str, profile_all_history_str, data_entry))
            
            logger.info(f"Dataset {ds_name}: Found {len(unique_profile_combinations)} unique profile combinations from {len(profile_texts)} samples")
            
            # Create embeddings for unique profile combinations
            # Get task-specific description for this dataset
            task_description = metadata[ds_name].get("task_description", None)
            user_profile_format = getattr(args, 'user_profile_format', 'history')
            
            # First, format all profile combinations to get the final text that will be embedded
            from hyper_llm_modulator.utils.preprocessing import (
                format_profile_text,
                apply_personalization_template,
                is_gen_profile_only_task,
            )
            if is_gen_profile_only_task(ds_name):
                if user_profile_format != "gen_profile":
                    logger.info(f"Dataset {ds_name}: forcing user_profile_format=gen_profile for embedding inputs")
                user_profile_format = "gen_profile"

            formatted_profile_texts = []
            combination_to_formatted = {}
            for profile_text_str, profile_all_history_str, data_entry in unique_profile_combinations:
                formatted_text = format_profile_text(profile_text_str, user_profile_format, profile_all_history_str, data_entry, profile_k, ds_name, include_history_stat)
                formatted_profile_texts.append(formatted_text)
                # Use the same combination key as used for deduplication
                if isinstance(data_entry, dict):
                    normalized_items = tuple(
                        sorted(
                            (
                                key,
                                json.dumps(value, sort_keys=True)
                                if isinstance(value, (dict, list))
                                else str(value),
                            )
                            for key, value in data_entry.items()
                        )
                    )
                else:
                    normalized_items = tuple()
                combination_key = (profile_text_str, profile_all_history_str, normalized_items)
                combination_to_formatted[combination_key] = formatted_text
            
            # Create a representation for caching that includes the combinations and format
            # Exclude 'history_stat' contents from the hash to keep cache keys compact and stable
            def _repr_without_hist_stat(combo):
                profile_text_str, profile_all_history_str, data_entry = combo
                if isinstance(data_entry, dict):
                    filtered_items = tuple(sorted((k, v) for k, v in data_entry.items() if k != "history_stat"))
                else:
                    filtered_items = tuple()
                return (profile_text_str, profile_all_history_str, filtered_items)

            combinations_repr = [_repr_without_hist_stat(combo) for combo in unique_profile_combinations]
            # Include history_stat flag in cache name only for lamp_movie and lamp_news_cat tasks
            history_stat_tag = ""
            if include_history_stat and ("lamp_movie" in ds_name or "lamp_news_cat" in ds_name):
                history_stat_tag = "_with_hist_stat"
            emb_tokenizer_path = emb_tokenizer.name_or_path.strip('/') if emb_tokenizer is not None else "no_emb_tokenizer"
            random_profile_marker = "random_profile_"
            random_profile_tag = "_random_profiles" if any(
                isinstance(text, str) and text.startswith(random_profile_marker)
                for text in formatted_profile_texts
            ) else ""
            ds_repr = (
                f"{ds_name}_profile_combinations_{user_profile_format}_{profile_k}_{task_description}_"
                f"{json.dumps(sorted(combinations_repr))}_{emb_tokenizer_path}{history_stat_tag}{random_profile_tag}"
            )
            ds_hash = hashlib.sha256(ds_repr.encode("utf-8")).hexdigest()
            os.makedirs(f"{EMBS_DIR}/", exist_ok=True)
            
            if glob(f"{EMBS_DIR}/{ds_hash}.pt"):
                logger.debug(f"Loading preprocessed profile combination embeddings: {ds_hash}")
                unique_profile_embs = torch.load(f"{EMBS_DIR}/{ds_hash}.pt", map_location="cpu")
                _update_hash_map(
                    EMBS_DIR,
                    f"{ds_hash}.pt",
                    ds_name,
                    extra={
                        "context": "profile_text_embs",
                        "emb_tokenizer": emb_tokenizer.name_or_path if emb_tokenizer is not None else "None",
                        "user_profile_format": user_profile_format,
                        "profile_k": profile_k,
                    },
                )
            else:
                # Apply personalization template to formatted texts for embedding
                if callable(task_desc_format_fn) and hasattr(task_desc_format_fn, 'func'):
                    # For personalization templates, apply directly to formatted texts (not lists)
                    templated_texts = [apply_personalization_template(text, task_description) for text in formatted_profile_texts]
                else:
                    # For non-personalization cases, apply template function directly
                    templated_texts = [task_desc_format_fn(text) for text in formatted_profile_texts]
                
                # Embed the final templated texts
                unique_profile_embs = embed_texts(
                    templated_texts,
                    emb_model,
                    emb_tokenizer,  # Always use emb_tokenizer for embedding generation
                    lambda x: x,  # Identity function since templating is already applied
                    pooling_fn,
                    device=emb_model.device,
                    batch_size=4,
                    use_api_embedding=use_api_embedding,
                    api_embedding_kwargs=api_embedding_kwargs,
                    max_token_per_profile=args.max_tokens_per_profile,
                ).to("cpu")
                
                torch.save(unique_profile_embs, f"{EMBS_DIR}/{ds_hash}.pt")
                _update_hash_map(
                    EMBS_DIR,
                    f"{ds_hash}.pt",
                    ds_name,
                    extra={
                        "context": "profile_text_embs",
                        "emb_tokenizer": emb_tokenizer.name_or_path if emb_tokenizer is not None else "None",
                        "user_profile_format": user_profile_format,
                        "profile_k": profile_k,
                    },
                )
            
            # Create mapping from formatted profile text to embedding (for lookup during training)
            profile_to_emb = {}
            for profile_combination, emb in zip(unique_profile_combinations, unique_profile_embs):
                profile_text_str, profile_all_history_str, data_entry = profile_combination
                formatted_key = format_profile_text(profile_text_str, user_profile_format, profile_all_history_str, data_entry, profile_k, ds_name, include_history_stat)
                profile_to_emb[formatted_key] = emb
            
            # Store individual profile combinations for each sample (in original order)
            profile_combinations_per_sample = []
            formatted_profiles_per_sample = []
            for i in range(len(profile_texts)):
                profile_text = profile_texts[i]
                profile_all_history = profile_all_histories[i]
                
                # Ensure both are strings
                profile_text_str = str(profile_text) if profile_text is not None else ""
                profile_all_history_str = str(profile_all_history) if profile_all_history is not None else ""
                
                # Create data entry for this sample
                data_entry = {}
                if profile_k > 0 and profile_retrieval_fields:
                    profile_retrieval_key = f"profile_retrieval_k{profile_k}"
                    if profile_retrieval_key in profile_retrieval_fields:
                        data_entry[profile_retrieval_key] = profile_retrieval_fields[profile_retrieval_key][i]

                # Attach history statistics if available
                if history_stats is not None:
                    try:
                        data_entry["history_stat"] = history_stats[i]
                    except Exception:
                        pass
                
                if isinstance(data_entry, dict):
                    normalized_items = tuple(
                        sorted(
                            (
                                key,
                                json.dumps(value, sort_keys=True)
                                if isinstance(value, (dict, list))
                                else str(value),
                            )
                            for key, value in data_entry.items()
                        )
                    )
                else:
                    normalized_items = tuple()
                combo_key = (profile_text_str, profile_all_history_str, normalized_items)

                profile_combinations_per_sample.append((profile_text_str, profile_all_history_str, data_entry))
                formatted_profiles_per_sample.append(combination_to_formatted.get(combo_key, format_profile_text(
                    profile_text_str,
                    user_profile_format,
                    profile_all_history_str,
                    data_entry,
                    profile_k,
                    ds_name,
                    include_history_stat,
                )))
            
            # Store the mapping and the per-sample profile combinations
            ds_embs_dict[ds_name] = {
                'profile_to_emb': profile_to_emb,
                'profile_lists': profile_combinations_per_sample,  # Now tuples of (profile_text, profile_all_history, data_entry)
                'user_profile_format': user_profile_format,
                'profile_k': profile_k,
                'include_history_stat': include_history_stat,
                'dataset_tokenizer_used': dataset_tokenizer.name_or_path,
                'embedding_tokenizer_used': emb_tokenizer.name_or_path,
                'formatted_profiles': formatted_profiles_per_sample,
            }
            
        else:
            logger.warning(f"Dataset {ds_name} does not contain profile_text field, falling back to descriptions")
            # Fallback to original description-based approach
            descriptions = metadata[ds_name]["descriptions"][: args.n_descs_per_ds]
            
            # Get task-specific description for this dataset
            task_description = metadata[ds_name].get("task_description", None)
            
            # Create task-specific template function for this dataset
            if callable(task_desc_format_fn) and hasattr(task_desc_format_fn, 'func'):
                # task_desc_format_fn is a partial function from create_personalization_template_fn
                dataset_task_desc_format_fn = task_desc_format_fn(task_description)
            else:
                # Fallback for non-personalization cases
                dataset_task_desc_format_fn = task_desc_format_fn
            
            task_embs = get_task_embs(
                {ds_name: descriptions},
                emb_model=emb_model,
                emb_tokenizer=emb_tokenizer,
                task_desc_format_fn=dataset_task_desc_format_fn,
                pooling_fn=pooling_fn,
                device=device,
                use_api_embedding=use_api_embedding,
                api_embedding_kwargs=api_embedding_kwargs,
            )
            ds_embs_dict[ds_name] = task_embs[ds_name]
    
    return ds_embs_dict


def get_profile_text_embs_dict(args, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, metadata, device):
    """Get embeddings dict using profile_text from each dataset instead of descriptions.
    
    Note: profile_text is expected to be a list of strings for each sample.
    We embed all unique profile texts and store them for sampling during training.
    
    WARNING: This function may have tokenizer consistency issues. Use get_profile_text_embs_dict_safe instead.
    """
    logger.warning("Using potentially unsafe profile text extraction. Consider using get_profile_text_embs_dict_safe instead.")
    return get_profile_text_embs_dict_safe(args, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, metadata, device, main_tokenizer=None, include_history_stat=getattr(args, 'include_history_stat', False))


def get_embs_dict(args, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, metadata, device, main_tokenizer=None, use_api_embedding=False, api_embedding_kwargs=None):
    # Check if datasets have profile_text field - if so, use that instead of descriptions
    has_profile_text = check_datasets_for_profile_text(ds_names, metadata)
    
    if has_profile_text:
        # Use profile_text from datasets instead of descriptions from metadata
        ds_embs_dict = get_profile_text_embs_dict_safe(
            args, emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, metadata, device, main_tokenizer,
            use_api_embedding=use_api_embedding, api_embedding_kwargs=api_embedding_kwargs,
            include_history_stat=getattr(args, 'include_history_stat', False)
        )
        return ds_embs_dict
    
    # Original logic for backward compatibility when profile_text is not available
    ds_descs_dict = {ds: metadata[ds]["descriptions"][: args.n_descs_per_ds] for ds in ds_names}

    ds_embs_dict = None
    if args.use_per_task_emb:
        # Process each dataset individually to use its specific task description
        ds_embs_dict = {}
        for ds_name in ds_names:
            # Get task-specific description for this dataset
            task_description = metadata[ds_name].get("task_description", None)
            
            # Create task-specific template function for this dataset
            if callable(task_desc_format_fn) and hasattr(task_desc_format_fn, 'func'):
                # task_desc_format_fn is a partial function from create_personalization_template_fn
                dataset_task_desc_format_fn = task_desc_format_fn(task_description)
            else:
                # Fallback for non-personalization cases
                dataset_task_desc_format_fn = task_desc_format_fn
            
            task_embs = get_task_embs(
                {ds_name: ds_descs_dict[ds_name]},
                emb_model=emb_model,
                emb_tokenizer=emb_tokenizer,
                task_desc_format_fn=dataset_task_desc_format_fn,
                pooling_fn=pooling_fn,
                device=device,
                use_api_embedding=use_api_embedding,
                api_embedding_kwargs=api_embedding_kwargs,
            )
            ds_embs_dict[ds_name] = task_embs[ds_name]
    elif args.use_inp_as_desc or args.use_default_desc or args.use_per_sample_desc:
        
        # the description has to be tokenized by emb_tokenizer not the base model's tokenizer
        ds_descs_dict = {
            ds_name: load_and_format_dataset(
                metadata,
                emb_tokenizer,
                args.sft_mode,
                is_intx_model=emb_tokenizer.chat_template is not None,
                ds_name=ds_name,
                ds_kwargs=metadata[ds_name]["ds_kwargs"],
            )
            for ds_name in ds_names
        }
        # use per sample description
        if args.use_per_sample_desc:
            ds_descs_dict = {ds_name: ds_descs_dict[ds_name]["context"] for ds_name in ds_names}
        else:
            # use input as description
            ds_descs_dict = {ds_name: ds_descs_dict[ds_name]["prompt"] for ds_name in ds_descs_dict}
            if args.use_default_desc:
                for ds_name in ds_descs_dict:
                    prompts = ds_descs_dict[ds_name]
                    for i in range(len(prompts)):
                        prompts[i] = prompts[i].split("\n\n")[0]
                    ds_descs_dict[ds_name] = prompts

        ds_embs_dict = get_inp_prompt_emb(
            emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, ds_descs_dict, metadata,
            use_api_embedding=use_api_embedding, api_embedding_kwargs=api_embedding_kwargs
        )
    else:
        # one-hot task indicator
        ds_embs_dict = {ds_name: None for ds_name in ds_names}

    logger.debug(f"{ds_embs_dict=}")
    return ds_embs_dict


@torch.no_grad()
def get_inp_prompt_emb(emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, ds_names, ds_descs_dict, metadata=None, use_api_embedding=False, api_embedding_kwargs=None):
    # Handle case when emb_model is None (e.g., in mt-lora mode)
    if emb_model is None:
        logger.info("emb_model is None in get_inp_prompt_emb, returning None embeddings")
        return {ds_name: None for ds_name in ds_names}
    
    ds_embs_dict = {}
    for ds_name in ds_names:
        # Get task-specific description for this dataset
        task_description = metadata[ds_name].get("task_description", None) if metadata else None
        
        # Create task-specific template function for this dataset
        if callable(task_desc_format_fn) and hasattr(task_desc_format_fn, 'func'):
            # task_desc_format_fn is a partial function from create_personalization_template_fn
            dataset_task_desc_format_fn = task_desc_format_fn(task_description)
        else:
            # Fallback for non-personalization cases
            dataset_task_desc_format_fn = task_desc_format_fn
        
        emb_tokenizer_path = emb_tokenizer.name_or_path.strip('/') if emb_tokenizer is not None else "no_emb_tokenizer"
        ds_repr = f"{ds_name}_{task_description}_{json.dumps(ds_descs_dict[ds_name])}_{emb_tokenizer_path}"
        ds_hash = hashlib.sha256(ds_repr.encode("utf-8")).hexdigest()
        os.makedirs(f"{EMBS_DIR}/", exist_ok=True)
        if glob(f"{EMBS_DIR}/{ds_hash}.pt"):
            logger.debug(f"Loading preprocessed dataset: {ds_hash}")
            ds_embs_dict[ds_name] = torch.load(f"{EMBS_DIR}/{ds_hash}.pt", map_location="cpu")
            _update_hash_map(
                EMBS_DIR,
                f"{ds_hash}.pt",
                ds_name,
                extra={
                    "context": "inp_prompt_emb",
                    "emb_tokenizer": emb_tokenizer_path,
                },
            )
        else:
            ds_embs_dict[ds_name] = embed_texts(
                ds_descs_dict[ds_name],
                emb_model,
                emb_tokenizer,
                dataset_task_desc_format_fn,  # Use dataset-specific template function
                pooling_fn,
                device=emb_model.device,
                batch_size=1,
                use_api_embedding=use_api_embedding,
                api_embedding_kwargs=api_embedding_kwargs,
            ).to("cpu")
            torch.save(ds_embs_dict[ds_name], f"{EMBS_DIR}/{ds_hash}.pt")
            _update_hash_map(
                EMBS_DIR,
                f"{ds_hash}.pt",
                ds_name,
                extra={
                    "context": "inp_prompt_emb",
                    "emb_tokenizer": emb_tokenizer_path,
                },
            )
    return ds_embs_dict


@torch.no_grad()
def get_recon_train_data(state_dict, target_modules, layer_indices, device, output_delta_w=False):
    layer_indices_out, lora_A, lora_B, target_deltaW = (
        defaultdict(list),
        {target_module: [None for _ in range(len(layer_indices))] for target_module in target_modules},
        {target_module: [None for _ in range(len(layer_indices))] for target_module in target_modules},
        dict(),
    )

    for k, v in state_dict.items():
        for target_module in target_modules:
            if target_module in k:
                layer_idx = int(k.split("layers.")[-1].split(".")[0])
                if layer_idx in layer_indices:
                    if "lora_A" in k:
                        lora_A[target_module][layer_idx] = v
                        layer_indices_out[target_module].append(layer_idx)
                    elif "lora_B" in k:
                        lora_B[target_module][layer_idx] = v

    for target_module in target_modules:
        lora_A[target_module] = torch.stack(lora_A[target_module], dim=0).to(device)
        lora_B[target_module] = torch.stack(lora_B[target_module], dim=0).to(device)
        if output_delta_w:
            target_deltaW[target_module] = (
                torch.bmm(
                    lora_B[target_module],
                    lora_A[target_module],
                )
                .to(torch.float32)
                .to(device)
            )

        layer_indices_out[target_module] = torch.tensor(
            sorted(layer_indices_out[target_module]),
            dtype=torch.long,
            device=device,
        )

    return dict(
        layer_indices=layer_indices_out,
        lora_A=lora_A,
        lora_B=lora_B,
        target_deltaW=target_deltaW,
    )


def test_local_dataset_loading():
    """Test function to validate local dataset loading functionality."""
    import tempfile
    import json
    
    # Test data
    test_data = [
        {"input": "What is 2+2?", "output": "4", "profile_text": ["I like math", "I prefer simple answers"]},
        {"input": "What is 3+3?", "output": "6", "profile_text": ["I enjoy calculations", "Step by step please"]},
        {"input": "What is 5*5?", "output": "25", "profile_text": ["I love multiplication", "Show your work"]},
    ]
    
    # Test JSONL loading
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        jsonl_path = f.name
    
    try:
        # Test loading JSONL file
        print("Testing JSONL loading...")
        ds_kwargs = {"path": jsonl_path, "split": "train[:2]"}
        dataset = load_dataset_with_local_support(**ds_kwargs)
        print(f"Loaded {len(dataset)} samples from JSONL")
        print(f"First sample: {dataset[0]}")
        
        # Test detection function
        dataset_type = detect_local_dataset_type(jsonl_path)
        print(f"Detected dataset type: {dataset_type}")
        
        print("Local dataset loading test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        # Cleanup
        os.unlink(jsonl_path)


if __name__ == "__main__":
    # Test local dataset loading
    test_local_dataset_loading()
    
    # Original test code
    from datasets import load_dataset

    seed = 42
    ds1 = load_dataset("Lots-of-LoRAs/task022_cosmosqa_passage_inappropriate_binary", "default", split="train[:5]")
    ds2 = load_dataset("Lots-of-LoRAs/task033_winogrande_answer_generation", split="train[:5]")
    ds3 = load_dataset("Lots-of-LoRAs/task034_winogrande_question_modification_object", split="train[:5]")
    ds4 = load_dataset("Lots-of-LoRAs/task035_winogrande_question_modification_person", split="train[:5]")
    dataset = ConcatDataset([ds1, ds2, ds3, ds4])
    sampler = HierachicalBatchSampler(dataset, 2, 2)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    breakpoint()
    for batch in repeat_iterator(dataloader):
        print(batch["id"])
        breakpoint()
    print("done")
