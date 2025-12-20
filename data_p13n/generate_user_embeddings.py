#!/usr/bin/env python3
"""
Generate user embeddings for all users across multiple datasets using Qwen3-Emb-4B model or API.

This script processes LaMP, LongLaMP, PRISM, OpinionQA, PersonalReddit, EC, and ALOE datasets,
extracts user profiles according to dataset-specific rules, and generates embeddings
using either:
1. Local Qwen3-Emb-4B model with the specified prompt template
2. API-based embedding models (OpenAI or vLLM)

Dataset Processing:
- LaMP: Split by 7 tasks (citation, movie, news_cat, news_headline, product, scholarly_title, tweet)
- LongLaMP: Split by 3 tasks (abstract_generation, product_review, topic_writing)
- PRISM/EC/OpinionQA/PersonalReddit/ALOE: Processed as single datasets

User Profile Extraction Rules:
- LaMP/LongLaMP: Extract from user history behavior using extract_recent_user_profile()
- PRISM/EC/OpinionQA/PersonalReddit: Use the profile_text field directly
- Alternatively: Use pre-generated profile files from generated_profile directory (supports both JSON and JSONL formats)

Output Files:
- LaMP: lamp_{task}_user_embeddings.npz (7 files)
- LongLaMP: longlamp_{task}_user_embeddings.npz (3 files)
- Others: {dataset}_user_embeddings.npz (5 files)

Prompt Template:
"Instruct: Focus on extracting personalization features and user preferences and behavior patterns from the following user profile\nUser Profile: {user_profile_text}"

Usage Examples:

1. Local model (default):
   python generate_user_embeddings.py --data_dir /path/to/datasets --output_dir ./embeddings

2. OpenAI API:
   python generate_user_embeddings.py --use_api --api_model text-embedding-3-large --api_key your_key

3. vLLM API (local server):
   python generate_user_embeddings.py --use_api --api_model Qwen/Qwen3-Embedding-4B --api_base http://localhost:8000

4. Process specific dataset with API:
   python generate_user_embeddings.py --use_api --api_model text-embedding-3-large --dataset LaMP --task movie

5. OpenAI with custom settings:
   python generate_user_embeddings.py --openai_model text-embedding-3-large --openai_api_key your_key --batch_size 50

6. vLLM with custom server:
   python generate_user_embeddings.py --vllm_model custom-embedding-model --vllm_api_base http://your-server:8000

7. Use pre-generated profiles with API:
   python generate_user_embeddings.py --use_generated_profiles --generated_profile_dir /path/to/generated_profile --use_api --api_model text-embedding-3-large

8. Use generated profiles for specific dataset/task:
   python generate_user_embeddings.py --use_generated_profiles --generated_profile_dir ./data_p13n/generated_profile --dataset LaMP --task movie --use_api --api_model text-embedding-3-large
"""

import json
import os
import torch
from pathlib import Path
from typing import Dict, List, Set, Optional
from tqdm import tqdm
import hashlib
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
from functools import partial
import time

# Import functions from create_hf_datasets.py
from create_hf_datasets import (
    extract_recent_user_profile,
    extract_recent_conversation_profile,
    detect_history_format,
    format_history_item
)

# Import API embedding functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'hyper_llm_modulator', 'utils'))
from api_embedding import embed_texts_api, create_api_embedding_kwargs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserEmbeddingGenerator:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", device="cuda", batch_size=32, 
                 use_multi_gpu=True, max_memory_gb=70, num_workers=4, force_regenerate=False,
                 # API parameters
                 use_api=False, api_key=None, api_base=None, api_model=None,
                 api_max_retries=3, api_timeout=60, api_max_tokens_per_text=30000,
                 # Generated profile parameters
                 use_generated_profiles=False, generated_profile_dir=None):
        """
        Initialize the user embedding generator with support for both local models and APIs.
        
        Args:
            model_name: HuggingFace model name for local embeddings (default: Qwen3-Embedding-4B)
            device: Device to run the local model on (can be specific like "cuda:6")
            batch_size: Batch size for embedding generation (increased default)
            use_multi_gpu: Whether to use multiple GPUs with DataParallel (local model only)
            max_memory_gb: Maximum memory to use per GPU in GB (local model only)
            num_workers: Number of workers for data loading/processing
            force_regenerate: Force regeneration even if output files already exist
            
            # API parameters
            use_api: Whether to use API-based embedding generation instead of local model
            api_key: API key for authentication (if None, uses environment variable)
            api_base: API base URL (None for OpenAI, custom URL for vLLM)
            api_model: API model name (e.g., "text-embedding-3-large" for OpenAI)
            api_max_retries: Maximum number of retries for failed API requests
            api_timeout: Request timeout in seconds for API calls
            api_max_tokens_per_text: Maximum tokens per individual text for API calls
            
            # Generated profile parameters
            use_generated_profiles: Whether to use pre-generated profile files instead of raw data
            generated_profile_dir: Path to directory containing generated profile files
        """
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.use_multi_gpu = use_multi_gpu
        self.max_memory_gb = max_memory_gb
        self.num_workers = num_workers
        self.force_regenerate = force_regenerate
        
        # API configuration
        self.use_api = use_api
        self.api_key = api_key
        self.api_base = api_base
        self.api_model = api_model or model_name
        self.api_max_retries = api_max_retries
        self.api_timeout = api_timeout
        self.api_max_tokens_per_text = api_max_tokens_per_text
        
        # Generated profile configuration
        self.use_generated_profiles = use_generated_profiles
        self.generated_profile_dir = generated_profile_dir
        
        # Prompt template as specified
        self.task_description = "Focus on extracting personalization features and user preferences and behavior patterns from the following user profile"
        
        if self.use_api:
            # API mode - no local model loading
            self.model = None
            self.tokenizer = None
            self.available_gpus = []
            logger.info(f"Initialized API embedding generator (model: {self.api_model})")
            if self.api_base:
                logger.info(f"Using custom API base: {self.api_base}")
            else:
                logger.info("Using OpenAI API")
        else:
            # Local model mode - existing initialization logic
            # Auto-detect available GPUs if using multi-GPU
            if use_multi_gpu and torch.cuda.is_available():
                self.available_gpus = list(range(torch.cuda.device_count()))
                logger.info(f"Available GPUs: {self.available_gpus}")
                # Filter out heavily used GPUs (>90% memory usage)
                free_gpus = []
                for gpu_id in self.available_gpus:
                    try:
                        torch.cuda.set_device(gpu_id)
                        memory_used = torch.cuda.memory_allocated(gpu_id)
                        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                        usage_percent = memory_used / memory_total * 100
                        if usage_percent < 90:  # Less than 90% used
                            free_gpus.append(gpu_id)
                        logger.info(f"GPU {gpu_id}: {usage_percent:.1f}% memory used")
                    except:
                        pass
                
                if free_gpus:
                    self.available_gpus = free_gpus
                    logger.info(f"Using GPUs with <90% memory usage: {self.available_gpus}")
                else:
                    logger.warning("No GPUs with <90% memory usage found, using all available GPUs")
            
            # Load tokenizer and model
            logger.info(f"Loading local model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            
            # Set primary device
            if isinstance(device, str) and device.startswith("cuda:"):
                primary_device = device
            else:
                primary_device = f"cuda:{self.available_gpus[0]}" if use_multi_gpu and self.available_gpus else device
            
            self.model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2"
            ).to(primary_device)
            
            # Use DataParallel for multi-GPU if requested and available
            if use_multi_gpu and len(self.available_gpus) > 1:
                logger.info(f"Using DataParallel across GPUs: {self.available_gpus}")
                self.model = torch.nn.DataParallel(self.model, device_ids=self.available_gpus)
                # Increase batch size for multi-GPU
                self.batch_size = batch_size * len(self.available_gpus)
                logger.info(f"Increased batch size to {self.batch_size} for multi-GPU setup")
            
            self.model.eval()
        
    def format_profile_with_prompt(self, user_profile_text: str) -> str:
        """Format user profile text with the specified prompt template."""
        return f"Instruct: {self.task_description}\nUser Profile: {user_profile_text.strip()}"
    
    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last token pooling for Qwen3 embeddings."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def embed_texts_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a single batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tensor of embeddings with shape (len(texts), embedding_dim)
        """
        # Tokenize
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, 
                              truncation=True, max_length=30000).to(self.model.device if not isinstance(self.model, torch.nn.DataParallel) else self.model.module.device)
        
        with torch.no_grad():
            # Get model output
            outputs = self.model(**inputs)
            
            # Handle DataParallel output
            if isinstance(self.model, torch.nn.DataParallel):
                last_hidden_state = outputs.last_hidden_state
            else:
                last_hidden_state = outputs.last_hidden_state
            
            # Use last token pooling as recommended for Qwen3
            batch_embeddings = self.last_token_pool(last_hidden_state, inputs['attention_mask'])
            
            # Normalize embeddings
            import torch.nn.functional as F
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            return batch_embeddings.cpu()
    
    def embed_texts(self, texts: List[str], save_checkpoint_every: int = 10000) -> torch.Tensor:
        """
        Generate embeddings for a list of texts using either local model or API.
        
        Args:
            texts: List of text strings to embed
            save_checkpoint_every: Save intermediate results every N batches (local model only)
            
        Returns:
            Tensor of embeddings with shape (len(texts), embedding_dim)
        """
        if self.use_api:
            return self.embed_texts_api(texts)
        else:
            return self.embed_texts_local(texts, save_checkpoint_every)
    
    def embed_texts_api(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings using API-based embedding service.
        
        Args:
            texts: List of text strings to embed (should already be formatted with prompt)
            
        Returns:
            Tensor of embeddings with shape (len(texts), embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts using API (model: {self.api_model})")
        
        start_time = time.time()
        
        # Use the API embedding function
        # Note: texts should already be formatted with the prompt template
        embeddings = embed_texts_api(
            texts=texts,
            model=self.api_model,
            api_key=self.api_key,
            api_base=self.api_base,
            device=self.device,
            batch_size=self.batch_size,
            max_retries=self.api_max_retries,
            timeout=self.api_timeout,
            max_tokens_per_text=self.api_max_tokens_per_text
        )
        
        total_time = time.time() - start_time
        logger.info(f"API embedding generation completed in {total_time:.2f} seconds ({len(texts)/total_time:.1f} samples/sec)")
        
        return embeddings
    
    def embed_texts_local(self, texts: List[str], save_checkpoint_every: int = 10000) -> torch.Tensor:
        """
        Generate embeddings for a list of texts using the local model with memory optimization.
        
        Args:
            texts: List of text strings to embed
            save_checkpoint_every: Save intermediate results every N batches to avoid memory issues
            
        Returns:
            Tensor of embeddings with shape (len(texts), embedding_dim)
        """
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings", total=total_batches):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = self.embed_texts_batch(batch_texts)
                embeddings.append(batch_embeddings)
                
                # Memory management: clear cache periodically
                if (i // self.batch_size) % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Progress logging
                if (i // self.batch_size) % 1000 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    samples_per_sec = i / elapsed
                    logger.info(f"Processed {i}/{len(texts)} samples ({samples_per_sec:.1f} samples/sec)")
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"CUDA OOM at batch {i//self.batch_size}, reducing batch size temporarily")
                torch.cuda.empty_cache()
                gc.collect()
                # Try with smaller batch
                for j in range(0, len(batch_texts), self.batch_size // 2):
                    mini_batch = batch_texts[j:j + self.batch_size // 2]
                    batch_embeddings = self.embed_texts_batch(mini_batch)
                    embeddings.append(batch_embeddings)
        
        # Final memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        total_time = time.time() - start_time
        logger.info(f"Local embedding generation completed in {total_time:.2f} seconds ({len(texts)/total_time:.1f} samples/sec)")
        
        return torch.cat(embeddings, dim=0)
    
    def process_file_chunk(self, file_path: str, dataset_name: str, task_name: str = None) -> Dict[str, str]:
        """Process a single file and extract user profiles."""
        user_profiles = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    user_id = data.get('user_id', '')
                    
                    if not user_id:
                        continue
                    
                    # Extract profile based on dataset type
                    if dataset_name.lower() in ['lamp', 'longlamp']:
                        profile_text = self.extract_lamp_user_profile(data)
                    else:
                        profile_text = self.extract_direct_profile_text(data)
                    
                    if profile_text:
                        user_profiles[user_id] = profile_text
                        
                except (json.JSONDecodeError, Exception):
                    continue
        
        return user_profiles

    def extract_lamp_user_profile(self, user_data: Dict, max_context_length: int = 28000) -> str:
        """Extract user profile from LaMP/LongLaMP user data using history."""
        history = user_data.get('history', [])
        if not history:
            return ""
        
        # Use the same function as in create_hf_datasets.py
        return extract_recent_user_profile(history, max_context_length)
    
    def extract_direct_profile_text(self, user_data: Dict) -> str:
        """Extract user profile from datasets that have direct profile_text field."""
        return user_data.get('profile_text', '').strip()
    
    def process_generated_profile_files(self, generated_profile_dir: str, dataset_name: str, task_name: str = None) -> Dict[str, str]:
        """
        Process pre-generated user profile files from the generated_profile directory.
        
        Args:
            generated_profile_dir: Path to the generated_profile directory
            dataset_name: Name of the dataset (LaMP, LongLaMP, PRISM, etc.)
            task_name: Specific task name for LaMP/LongLaMP (e.g., 'movie', 'citation')
            
        Returns:
            Dictionary mapping user_id to user profile text
        """
        user_profiles = {}
        
        # Pattern matching for different file formats
        if dataset_name.lower() == 'lamp':
            if task_name:
                # LaMP with specific task: LaMP_processed_{task}_profiles.json
                profile_patterns = [
                    f"LaMP_processed_{task_name}_profiles.json",
                    # f"LaMP_processed_{task_name}_data_temp_profiles.jsonl"
                ]
            else:
                # All LaMP tasks
                profile_patterns = [
                    "LaMP_processed_*_profiles.json",
                    # "LaMP_processed_*_data_temp_profiles.jsonl"
                ]
        elif dataset_name.lower() == 'longlamp':
            if task_name:
                # LongLaMP with specific task: LongLaMP_{task}_profiles.json
                profile_patterns = [
                    f"LongLaMP_{task_name}_profiles.json",
                    # f"LongLaMP_{task_name}_data_temp_profiles.jsonl"
                ]
            else:
                # All LongLaMP tasks
                profile_patterns = [
                    "LongLaMP_*_profiles.json",
                    # "LongLaMP_*_data_temp_profiles.jsonl"
                ]
        elif dataset_name.lower() == 'prism':
            profile_patterns = [
                "PRISM_profiles.json",
                # "PRISM_temp_profiles.jsonl"  # Exclude temp profiles
            ]
        else:
            # Generic pattern for other datasets
            profile_patterns = [
                f"{dataset_name}_profiles.json",
                # f"{dataset_name}_temp_profiles.jsonl",
                f"{dataset_name.lower()}_profiles.json",
                # f"{dataset_name.lower()}_temp_profiles.jsonl"
            ]
        
        # Find matching files
        import glob
        found_files = []
        for pattern in profile_patterns:
            matching_files = glob.glob(os.path.join(generated_profile_dir, pattern))
            found_files.extend(matching_files)
        
        if not found_files:
            logger.warning(f"No generated profile files found for {dataset_name}" + 
                          (f" task {task_name}" if task_name else "") + 
                          f" in {generated_profile_dir}")
            logger.info(f"Searched for patterns: {profile_patterns}")
            return user_profiles
        
        logger.info(f"Found {len(found_files)} generated profile files for {dataset_name}" +
                   (f" task {task_name}" if task_name else "") + f": {[os.path.basename(f) for f in found_files]}")
        
        # Process each found file
        for file_path in found_files:
            try:
                if file_path.endswith('.jsonl'):
                    # JSONL format: each line has custom_id and content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                custom_id = data.get('custom_id', '')
                                content = data.get('content', '').strip()
                                
                                if custom_id and content:
                                    # Extract user_id from custom_id (e.g., "PRISM_PRISM_user9_5" -> "user9_5")
                                    user_id = self.extract_user_id_from_custom_id(custom_id, dataset_name)
                                    if user_id:
                                        user_profiles[user_id] = content
                                        
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON on line {line_num + 1} in {file_path}: {e}")
                                continue
                
                elif file_path.endswith('.json'):
                    # JSON format: data object with user_id -> profile_text mappings
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            
                            # Handle different JSON structures
                            profiles_data = None
                            if 'data' in data:
                                # LaMP format: {"data": {"lamp_8000003": "profile_text", ...}}
                                profiles_data = data['data']
                            elif 'prism_data' in data:
                                # PRISM format: {"prism_data": {"PRISM_PRISM_user9": "profile_text", ...}}
                                profiles_data = data['prism_data']
                            else:
                                # Direct format: {"user_id": "profile_text", ...}
                                profiles_data = data
                            
                            if profiles_data:
                                for user_id, profile_text in profiles_data.items():
                                    # Ensure profile_text is a string and not empty
                                    if user_id and isinstance(profile_text, str) and profile_text.strip():
                                        user_profiles[user_id] = profile_text.strip()
                                    elif user_id and profile_text:
                                        logger.warning(f"Skipping non-string profile for user {user_id}: {type(profile_text)}")
                                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON file {file_path}: {e}")
                            continue
                
                logger.info(f"Loaded {len(user_profiles)} user profiles from {os.path.basename(file_path)}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        logger.info(f"Total user profiles loaded for {dataset_name}" +
                   (f" task {task_name}" if task_name else "") + f": {len(user_profiles)}")
        return user_profiles
    
    def extract_user_id_from_custom_id(self, custom_id: str, dataset_name: str) -> str:
        """
        Extract user_id from custom_id based on dataset patterns.
        
        Args:
            custom_id: Custom ID like "PRISM_PRISM_user9_5" or "LaMP_movie_user123"
            dataset_name: Dataset name for context
            
        Returns:
            Extracted user_id or the custom_id itself if no pattern matches
        """
        if not custom_id:
            return ""
        
        # Common patterns:
        # PRISM: "PRISM_PRISM_user9_5" -> "user9_5" or "PRISM_user9_5"
        # LaMP: "LaMP_movie_user123" -> "user123" or "LaMP_movie_user123"
        
        if dataset_name.lower() == 'prism':
            # For PRISM, extract the last part after the second underscore
            parts = custom_id.split('_')
            if len(parts) >= 3:
                return '_'.join(parts[2:])  # "user9_5"
            else:
                return custom_id
        
        elif dataset_name.lower() in ['lamp', 'longlamp']:
            # For LaMP/LongLaMP, keep the original custom_id or extract user part
            if 'user' in custom_id.lower():
                # Try to extract user part
                parts = custom_id.split('_')
                user_parts = [part for part in parts if 'user' in part.lower()]
                if user_parts:
                    return user_parts[0]  # Take first user part
            return custom_id
        
        else:
            # For other datasets, use custom_id as-is or try to extract user part
            if 'user' in custom_id.lower():
                parts = custom_id.split('_')
                user_parts = [part for part in parts if 'user' in part.lower()]
                if user_parts:
                    return user_parts[0]
            return custom_id
    
    def process_dataset_files(self, dataset_dir: str, dataset_name: str, task_name: str = None) -> Dict[str, str]:
        """
        Process all files in a dataset directory and extract user profiles with parallel processing.
        
        Args:
            dataset_dir: Path to dataset directory
            dataset_name: Name of the dataset (LaMP, LongLaMP, PRISM, etc.)
            task_name: Specific task name for LaMP/LongLaMP (e.g., 'movie', 'citation')
            
        Returns:
            Dictionary mapping user_id to user profile text
        """
        # Find all relevant data files
        data_files = []
        for file in os.listdir(dataset_dir):
            if file.endswith('_data.jsonl'):
                # For LaMP/LongLaMP, filter by task name if specified
                if task_name and (dataset_name.lower() in ['lamp', 'longlamp']):
                    if task_name in file:
                        data_files.append(os.path.join(dataset_dir, file))
                else:
                    # For other datasets, include all data files
                    data_files.append(os.path.join(dataset_dir, file))
        
        task_info = f" (task: {task_name})" if task_name else ""
        logger.info(f"Processing {dataset_name}{task_info} dataset with files: {[os.path.basename(f) for f in data_files]}")
        
        all_user_profiles = {}
        
        # Process files in parallel
        if len(data_files) > 1 and self.num_workers > 1:
            logger.info(f"Processing {len(data_files)} files in parallel with {self.num_workers} workers")
            process_func = partial(self.process_file_chunk, dataset_name=dataset_name, task_name=task_name)
            
            with ThreadPoolExecutor(max_workers=min(self.num_workers, len(data_files))) as executor:
                results = list(executor.map(process_func, data_files))
            
            # Merge results
            for file_profiles in results:
                all_user_profiles.update(file_profiles)
        else:
            # Process files sequentially for small number of files
            for file_path in data_files:
                logger.info(f"Processing file: {os.path.basename(file_path)}")
                file_profiles = self.process_file_chunk(file_path, dataset_name, task_name)
                all_user_profiles.update(file_profiles)
        
        task_info = f" {task_name}" if task_name else ""
        logger.info(f"Extracted profiles for {len(all_user_profiles)} users from {dataset_name}{task_info}")
        return all_user_profiles
    
    def save_embeddings_chunked(self, user_ids: List[str], user_profiles: Dict[str, str], 
                               embeddings: torch.Tensor, output_file: str, dataset_name: str):
        """Save embeddings with memory-efficient chunked approach and robust file handling."""
        try:
            # Get the actual model name used for embedding generation
            actual_model_name = self.api_model if self.use_api else self.model_name
            
            # Ensure embeddings are on CPU before converting to numpy
            if embeddings.is_cuda:
                embeddings = embeddings.cpu()
            
            # Create output data
            output_data = {
                'dataset_name': dataset_name,
                'model_name': actual_model_name,
                'embedding_mode': 'api' if self.use_api else 'local',
                'user_ids': user_ids,
                'user_profiles': user_profiles,
                'embeddings': embeddings.numpy(),
                'embedding_dim': int(embeddings.shape[1]),
                'num_users': len(user_ids)
            }
            
            # Add API-specific metadata if using API mode
            if self.use_api:
                output_data.update({
                    'api_base': self.api_base,
                    'api_batch_size': self.batch_size,
                    'api_max_tokens_per_text': self.api_max_tokens_per_text
                })
            
            # Create temporary file first to ensure atomic write
            # Note: np.savez_compressed automatically adds .npz extension, so we need to account for this
            temp_base = output_file.replace('.npz', '.tmp')
            temp_output_file = temp_base + '.npz'  # This will be created by np.savez_compressed
            
            # Save with compression to temporary file
            logger.info(f"Saving embeddings to temporary file: {temp_output_file}")
            logger.debug(f"Using temp_base for np.savez_compressed: {temp_base}")
            
            try:
                np.savez_compressed(temp_base, **output_data)
            except Exception as save_error:
                logger.error(f"Failed to save with np.savez_compressed: {save_error}")
                raise
            
            # Small delay to ensure filesystem consistency
            import time
            time.sleep(0.1)
            
            # Verify the temporary file was written correctly
            if not os.path.exists(temp_output_file):
                # Check if file exists with any related name patterns
                import glob
                pattern_dir = os.path.dirname(temp_output_file)
                basename = os.path.basename(temp_base)
                related_files = glob.glob(os.path.join(pattern_dir, f"{basename}*"))
                logger.error(f"Expected temporary file not found: {temp_output_file}")
                logger.error(f"Files matching pattern {basename}*: {related_files}")
                raise FileNotFoundError(f"Temporary file was not created: {temp_output_file}")
            
            # Check file size is reasonable (should be > 1KB for any real dataset)
            temp_file_size = os.path.getsize(temp_output_file)
            if temp_file_size < 1024:
                raise ValueError(f"Output file suspiciously small ({temp_file_size} bytes): {temp_output_file}")
            
            # Atomic move from temp to final location
            logger.info(f"Moving temporary file to final location: {output_file}")
            os.rename(temp_output_file, output_file)
            
            # Verify final file exists and has correct size
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Final output file was not created: {output_file}")
            
            final_file_size = os.path.getsize(output_file)
            if final_file_size != temp_file_size:
                raise ValueError(f"File size changed during move: expected {temp_file_size}, got {final_file_size}")
            
            logger.info(f"✓ Successfully saved embeddings for {dataset_name} to {output_file} ({final_file_size} bytes)")
            
            # Try to verify we can load the file (quick sanity check)
            try:
                test_load = np.load(output_file)
                saved_embeddings_shape = test_load['embeddings'].shape
                expected_shape = (len(user_ids), embeddings.shape[1])
                if saved_embeddings_shape != expected_shape:
                    logger.warning(f"Saved embeddings shape {saved_embeddings_shape} != expected {expected_shape}")
                else:
                    logger.info(f"✓ Verified saved embeddings shape: {saved_embeddings_shape}")
                test_load.close()
            except Exception as e:
                logger.warning(f"Could not verify saved file integrity: {e}")
            
        except Exception as e:
            # Clean up temporary file if it exists
            temp_base = output_file.replace('.npz', '.tmp')
            temp_output_file = temp_base + '.npz'
            if os.path.exists(temp_output_file):
                try:
                    os.remove(temp_output_file)
                    logger.info(f"Cleaned up temporary file: {temp_output_file}")
                except:
                    pass
            
            logger.error(f"Failed to save embeddings for {dataset_name}: {e}")
            raise
        finally:
            # Clear from memory
            if 'output_data' in locals():
                del output_data
            gc.collect()

    def generate_embeddings_for_dataset(self, dataset_dir: str, dataset_name: str, output_dir: str, task_name: str = None):
        """Generate embeddings for all users in a dataset or specific task with optimization."""
        task_info = f" (task: {task_name})" if task_name else ""
        logger.info(f"Processing {dataset_name}{task_info} dataset...")
        
        # Create output directory early to check for existing files
        os.makedirs(output_dir, exist_ok=True)
        
        # Create file names with task suffix for LaMP/LongLaMP
        if task_name:
            file_prefix = f"{dataset_name.lower()}_{task_name}"
            full_dataset_name = f"{dataset_name}_{task_name}"
        else:
            file_prefix = f"{dataset_name.lower()}"
            full_dataset_name = dataset_name
        
        # Check if output file already exists (unless force regenerate is enabled)
        output_file = os.path.join(output_dir, f"{file_prefix}_user_embeddings.npz")
        if os.path.exists(output_file) and not self.force_regenerate:
            logger.info(f"✓ Embedding file already exists, skipping {full_dataset_name}: {output_file}")
            
            # Try to load existing file to get metadata for return value
            try:
                existing_data = np.load(output_file)
                num_users = existing_data.get('num_users', 0)
                if num_users == 0 and 'user_ids' in existing_data:
                    num_users = len(existing_data['user_ids'])
                # Convert numpy scalar to Python int to avoid JSON serialization issues
                num_users = int(num_users) if hasattr(num_users, 'item') else num_users
                existing_data.close()
                
                # Also check for metadata file
                metadata_file = os.path.join(output_dir, f"{file_prefix}_user_metadata.json")
                
                return {
                    'dataset_name': full_dataset_name,
                    'num_users': num_users,
                    'output_file': output_file,
                    'metadata_file': metadata_file,
                    'processing_time': 0.0,
                    'skipped': True
                }
            except Exception as e:
                logger.warning(f"Could not read existing embedding file {output_file}: {e}")
                logger.info(f"Will regenerate embeddings for {full_dataset_name}")
        elif os.path.exists(output_file) and self.force_regenerate:
            logger.info(f"✓ Force regenerate enabled, will overwrite existing file: {output_file}")
        
        # Extract user profiles with fallback mechanism
        user_profiles = {}
        
        if self.use_generated_profiles and self.generated_profile_dir:
            # Try to load generated profiles first
            user_profiles = self.process_generated_profile_files(self.generated_profile_dir, dataset_name, task_name)
            
            if not user_profiles:
                logger.warning(f"No generated profiles found for {dataset_name}{task_info}, falling back to original data files")
                # Fallback to original dataset files
                user_profiles = self.process_dataset_files(dataset_dir, dataset_name, task_name)
            else:
                logger.info(f"✓ Using generated profiles for {dataset_name}{task_info}")
        else:
            # Use original dataset files
            user_profiles = self.process_dataset_files(dataset_dir, dataset_name, task_name)
        
        if not user_profiles:
            logger.warning(f"No user profiles found for {dataset_name}{task_info} (tried both generated and original data)")
            return None
        
        # Format profiles with prompt template
        user_ids = list(user_profiles.keys())
        formatted_profiles = [self.format_profile_with_prompt(user_profiles[user_id]) 
                            for user_id in user_ids]
        
        logger.info(f"Generating embeddings for {len(user_ids)} users in {dataset_name}{task_info}...")
        
        # Generate embeddings with optimization
        start_time = time.time()
        embeddings = self.embed_texts(formatted_profiles)
        embedding_time = time.time() - start_time
        
        logger.info(f"Embedding generation took {embedding_time:.2f} seconds ({len(user_ids)/embedding_time:.1f} users/sec)")
        
        # Save embeddings using optimized method
        self.save_embeddings_chunked(user_ids, user_profiles, embeddings, output_file, full_dataset_name)
        
        # Also save human-readable JSON with user profiles (without embeddings)
        metadata_file = os.path.join(output_dir, f"{file_prefix}_user_metadata.json")
        
        # Get the actual model name used for embedding generation
        actual_model_name = self.api_model if self.use_api else self.model_name
        
        metadata = {
            'dataset_name': full_dataset_name,
            'model_name': actual_model_name,
            'embedding_mode': 'api' if self.use_api else 'local',
            'num_users': len(user_ids),
            'embedding_dim': int(embeddings.shape[1]),
            'user_profiles': user_profiles,
            'processing_time_seconds': embedding_time,
            'users_per_second': len(user_ids) / embedding_time
        }
        
        # Add API-specific metadata if using API mode
        if self.use_api:
            metadata.update({
                'api_base': self.api_base,
                'api_batch_size': self.batch_size,
                'api_max_tokens_per_text': self.api_max_tokens_per_text,
                'api_max_retries': self.api_max_retries,
                'api_timeout': self.api_timeout
            })
        
        # Save metadata with robust file handling
        try:
            temp_metadata_file = metadata_file + '.tmp'
            logger.info(f"Saving metadata to temporary file: {temp_metadata_file}")
            
            with open(temp_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Verify temporary file was created and has reasonable size
            if not os.path.exists(temp_metadata_file):
                raise FileNotFoundError(f"Temporary metadata file was not created: {temp_metadata_file}")
            
            temp_file_size = os.path.getsize(temp_metadata_file)
            if temp_file_size < 100:  # JSON should be at least 100 bytes
                raise ValueError(f"Metadata file suspiciously small ({temp_file_size} bytes): {temp_metadata_file}")
            
            # Atomic move to final location
            logger.info(f"Moving temporary metadata file to final location: {metadata_file}")
            os.rename(temp_metadata_file, metadata_file)
            
            # Verify final file
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Final metadata file was not created: {metadata_file}")
            
            final_file_size = os.path.getsize(metadata_file)
            logger.info(f"✓ Successfully saved metadata for {full_dataset_name} to {metadata_file} ({final_file_size} bytes)")
            
            # Quick verification that we can load the JSON
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    test_metadata = json.load(f)
                if test_metadata.get('num_users') != len(user_ids):
                    logger.warning(f"Metadata verification failed: expected {len(user_ids)} users, got {test_metadata.get('num_users')}")
                else:
                    logger.info(f"✓ Verified metadata file integrity")
            except Exception as e:
                logger.warning(f"Could not verify metadata file integrity: {e}")
                
        except Exception as e:
            # Clean up temporary file if it exists
            temp_metadata_file = metadata_file + '.tmp'
            if os.path.exists(temp_metadata_file):
                try:
                    os.remove(temp_metadata_file)
                    logger.info(f"Cleaned up temporary metadata file: {temp_metadata_file}")
                except:
                    pass
            
            logger.error(f"Failed to save metadata for {full_dataset_name}: {e}")
            raise
        
        # Clean up memory
        del embeddings, formatted_profiles
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'dataset_name': full_dataset_name,
            'num_users': len(user_ids),
            'output_file': output_file,
            'metadata_file': metadata_file,
            'processing_time': embedding_time
        }
    
    def generate_all_embeddings(self, data_root_dir: str, output_dir: str):
        """Generate embeddings for all datasets."""
        datasets = {
            'LaMP': 'LaMP',
            'LongLaMP': 'LongLaMP', 
            'PRISM': 'PRISM',
            'OpinionQA': 'OpinionQA',
            'PersonalReddit': 'PersonalReddit',
            'EC': 'EC',
            'ALOE': 'ALOE'
        }
        
        # Define tasks for LaMP and LongLaMP
        lamp_tasks = ['citation', 'movie', 'news_cat', 'news_headline', 'product', 'scholarly_title', 'tweet']
        longlamp_tasks = ['abstract_generation', 'product_review', 'topic_writing']
        
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overall metadata
        overall_metadata = {
            'model_name': self.api_model if self.use_api else self.model_name,
            'embedding_mode': 'api' if self.use_api else 'local',
            'task_description': self.task_description,
            'datasets_processed': [],
            'total_users': 0,
            'successful_saves': [],
            'skipped_datasets': [],
            'failed_datasets': []
        }
        
        # Add API-specific metadata if using API mode
        if self.use_api:
            overall_metadata.update({
                'api_base': self.api_base,
                'api_batch_size': self.batch_size,
                'api_max_tokens_per_text': self.api_max_tokens_per_text,
                'api_max_retries': self.api_max_retries,
                'api_timeout': self.api_timeout
            })
        
        for dataset_name, dataset_dir_name in datasets.items():
            dataset_path = os.path.join(data_root_dir, dataset_dir_name)
            
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset directory not found: {dataset_path}")
                overall_metadata['failed_datasets'].append({
                    'dataset': dataset_name,
                    'reason': 'Directory not found',
                    'path': dataset_path
                })
                continue
            
            # Process LaMP and LongLaMP by individual tasks
            if dataset_name == 'LaMP':
                for task in lamp_tasks:
                    try:
                        logger.info(f"Processing LaMP task: {task}")
                        result = self.generate_embeddings_for_dataset(dataset_path, dataset_name, output_dir, task)
                        if result:
                            if result.get('skipped', False):
                                # File already existed, was skipped
                                logger.info(f"✓ Skipped {result['dataset_name']} (file already exists)")
                                overall_metadata['skipped_datasets'].append(result['dataset_name'])
                            else:
                                # Ensure files are saved and log success
                                logger.info(f"✓ Successfully saved {result['dataset_name']} embeddings to {result['output_file']}")
                                logger.info(f"✓ Successfully saved {result['dataset_name']} metadata to {result['metadata_file']}")
                                overall_metadata['successful_saves'].append(result['dataset_name'])
                                
                                # Force filesystem sync to ensure files are written
                                os.sync() if hasattr(os, 'sync') else None
                            
                            # Add to processed datasets regardless of whether skipped or newly generated
                            overall_metadata['datasets_processed'].append({
                                'dataset_name': result['dataset_name'],
                                'num_users': result['num_users'],
                                'output_file': result['output_file'],
                                'metadata_file': result['metadata_file'],
                                'processing_time': result['processing_time'],
                                'skipped': result.get('skipped', False)
                            })
                            overall_metadata['total_users'] += result['num_users']
                        else:
                            logger.warning(f"Failed to process LaMP task: {task}")
                            overall_metadata['failed_datasets'].append({
                                'dataset': f"LaMP_{task}",
                                'reason': 'Processing returned None',
                                'path': dataset_path
                            })
                    except Exception as e:
                        logger.error(f"Error processing LaMP task {task}: {e}")
                        overall_metadata['failed_datasets'].append({
                            'dataset': f"LaMP_{task}",
                            'reason': str(e),
                            'path': dataset_path
                        })
                        continue
            
            elif dataset_name == 'LongLaMP':
                for task in longlamp_tasks:
                    try:
                        logger.info(f"Processing LongLaMP task: {task}")
                        result = self.generate_embeddings_for_dataset(dataset_path, dataset_name, output_dir, task)
                        if result:
                            if result.get('skipped', False):
                                # File already existed, was skipped
                                logger.info(f"✓ Skipped {result['dataset_name']} (file already exists)")
                                overall_metadata['skipped_datasets'].append(result['dataset_name'])
                            else:
                                # Ensure files are saved and log success
                                logger.info(f"✓ Successfully saved {result['dataset_name']} embeddings to {result['output_file']}")
                                logger.info(f"✓ Successfully saved {result['dataset_name']} metadata to {result['metadata_file']}")
                                overall_metadata['successful_saves'].append(result['dataset_name'])
                                
                                # Force filesystem sync to ensure files are written
                                os.sync() if hasattr(os, 'sync') else None
                            
                            # Add to processed datasets regardless of whether skipped or newly generated
                            overall_metadata['datasets_processed'].append({
                                'dataset_name': result['dataset_name'],
                                'num_users': result['num_users'],
                                'output_file': result['output_file'],
                                'metadata_file': result['metadata_file'],
                                'processing_time': result['processing_time'],
                                'skipped': result.get('skipped', False)
                            })
                            overall_metadata['total_users'] += result['num_users']
                        else:
                            logger.warning(f"Failed to process LongLaMP task: {task}")
                            overall_metadata['failed_datasets'].append({
                                'dataset': f"LongLaMP_{task}",
                                'reason': 'Processing returned None',
                                'path': dataset_path
                            })
                    except Exception as e:
                        logger.error(f"Error processing LongLaMP task {task}: {e}")
                        overall_metadata['failed_datasets'].append({
                            'dataset': f"LongLaMP_{task}",
                            'reason': str(e),
                            'path': dataset_path
                        })
                        continue
            
            else:
                # Process other datasets as single units
                try:
                    result = self.generate_embeddings_for_dataset(dataset_path, dataset_name, output_dir)
                    if result:
                        if result.get('skipped', False):
                            # File already existed, was skipped
                            logger.info(f"✓ Skipped {result['dataset_name']} (file already exists)")
                            overall_metadata['skipped_datasets'].append(result['dataset_name'])
                        else:
                            # Ensure files are saved and log success
                            logger.info(f"✓ Successfully saved {result['dataset_name']} embeddings to {result['output_file']}")
                            logger.info(f"✓ Successfully saved {result['dataset_name']} metadata to {result['metadata_file']}")
                            overall_metadata['successful_saves'].append(result['dataset_name'])
                            
                            # Force filesystem sync to ensure files are written
                            os.sync() if hasattr(os, 'sync') else None
                        
                        # Add to processed datasets regardless of whether skipped or newly generated
                        overall_metadata['datasets_processed'].append({
                            'dataset_name': result['dataset_name'],
                            'num_users': result['num_users'],
                            'output_file': result['output_file'],
                            'metadata_file': result['metadata_file'],
                            'processing_time': result['processing_time'],
                            'skipped': result.get('skipped', False)
                        })
                        overall_metadata['total_users'] += result['num_users']
                    else:
                        logger.warning(f"Failed to process dataset: {dataset_name}")
                        overall_metadata['failed_datasets'].append({
                            'dataset': dataset_name,
                            'reason': 'Processing returned None',
                            'path': dataset_path
                        })
                except Exception as e:
                    logger.error(f"Error processing {dataset_name}: {e}")
                    overall_metadata['failed_datasets'].append({
                        'dataset': dataset_name,
                        'reason': str(e),
                        'path': dataset_path
                    })
                    continue
        
        # Save overall metadata
        overall_metadata_file = os.path.join(output_dir, 'overall_metadata.json')
        with open(overall_metadata_file, 'w') as f:
            json.dump(overall_metadata, f, indent=2)
        
        logger.info(f"Completed processing all datasets. Total users: {overall_metadata['total_users']}")
        logger.info(f"Overall metadata saved to: {overall_metadata_file}")
        
        # Print detailed summary of generated files
        logger.info("\n=== Generated Files Summary ===")
        logger.info(f"Successfully processed: {len(overall_metadata['successful_saves'])}")
        logger.info(f"Skipped (already existed): {len(overall_metadata['skipped_datasets'])}")
        
        for dataset_info in overall_metadata['datasets_processed']:
            if dataset_info.get('skipped', False):
                status_icon = "⊖"  # Skipped
                status_text = f"skipped ({dataset_info['processing_time']:.2f}s)"
            else:
                status_icon = "✓"  # Newly processed
                status_text = f"processed ({dataset_info['processing_time']:.2f}s)"
            
            logger.info(f"  {status_icon} {dataset_info['dataset_name']}: {dataset_info['num_users']} users ({status_text})")
            logger.info(f"    - Embeddings: {dataset_info['output_file']}")
            logger.info(f"    - Metadata: {dataset_info['metadata_file']}")
        
        if overall_metadata['skipped_datasets']:
            logger.info(f"\nSkipped datasets (files already existed):")
            for skipped in overall_metadata['skipped_datasets']:
                logger.info(f"  ⊖ {skipped}")
        
        if overall_metadata['failed_datasets']:
            logger.info(f"\nFailed to process: {len(overall_metadata['failed_datasets'])}")
            for failed in overall_metadata['failed_datasets']:
                logger.info(f"  ✗ {failed['dataset']}: {failed['reason']}")
        
        logger.info(f"\nTotal datasets/tasks processed: {len(overall_metadata['datasets_processed'])}")
        logger.info(f"  - Newly generated: {len(overall_metadata['successful_saves'])}")
        logger.info(f"  - Skipped (existed): {len(overall_metadata['skipped_datasets'])}")
        logger.info(f"  - Failed: {len(overall_metadata['failed_datasets'])}")
        logger.info(f"Total users across all: {overall_metadata['total_users']}")


def cleanup_temporary_files(output_dir: str):
    """
    Clean up any incorrectly named temporary files (with .npz.tmp.npz extension).
    This fixes files created before the numpy extension handling was corrected.
    """
    import glob
    
    logger.info(f"Checking for incorrectly named temporary files in {output_dir}")
    
    # Find files with double .npz extension pattern
    temp_pattern = os.path.join(output_dir, "*.npz.tmp.npz")
    temp_files = glob.glob(temp_pattern)
    
    if not temp_files:
        logger.info("No incorrectly named temporary files found")
        return
    
    logger.info(f"Found {len(temp_files)} incorrectly named temporary files")
    
    for temp_file in temp_files:
        # Extract the correct final filename
        final_file = temp_file.replace('.npz.tmp.npz', '.npz')
        
        try:
            # Verify the temporary file is valid before renaming
            test_data = np.load(temp_file)
            num_users = test_data.get('num_users', len(test_data['user_ids']) if 'user_ids' in test_data else 0)
            # Convert numpy scalar to Python int
            num_users = int(num_users) if hasattr(num_users, 'item') else num_users
            test_data.close()
            
            if num_users > 0:
                logger.info(f"Renaming valid temporary file: {os.path.basename(temp_file)} -> {os.path.basename(final_file)} ({num_users} users)")
                os.rename(temp_file, final_file)
            else:
                logger.warning(f"Temporary file appears empty, removing: {temp_file}")
                os.remove(temp_file)
                
        except Exception as e:
            logger.error(f"Failed to process temporary file {temp_file}: {e}")
            logger.info(f"Consider manually checking/removing: {temp_file}")


def main():
    """Main function to generate user embeddings for all datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate user embeddings for personalization datasets")
    parser.add_argument("--data_dir", type=str, default=".", 
                       help="Root directory containing dataset folders")
    parser.add_argument("--output_dir", type=str, default="user_embeddings",
                       help="Output directory for user embeddings")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-4B",
                       help="HuggingFace model name for local embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run the local model on (e.g., 'cuda', 'cuda:6')")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Base batch size for embedding generation (will be multiplied by num_gpus if using multi-GPU)")
    parser.add_argument("--use_multi_gpu", action="store_true", default=True,
                       help="Use multiple GPUs with DataParallel (local model only)")
    parser.add_argument("--no_multi_gpu", action="store_true",
                       help="Disable multi-GPU usage")
    parser.add_argument("--max_memory_gb", type=int, default=70,
                       help="Maximum memory to use per GPU in GB (local model only)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for parallel file processing")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Process only specific dataset (LaMP, LongLaMP, PRISM, etc.)")
    parser.add_argument("--task", type=str, default=None,
                       help="Process only specific task (for LaMP/LongLaMP)")
    parser.add_argument("--force_regenerate", action="store_true",
                       help="Force regeneration of embeddings even if output files already exist")
    parser.add_argument("--cleanup_temp_files", action="store_true",
                       help="Clean up incorrectly named temporary files and exit")
    
    # API-related arguments
    parser.add_argument("--use_api", action="store_true",
                       help="Use API-based embedding generation instead of local model")
    parser.add_argument("--api_model", type=str, default=None,
                       help="API model name (e.g., 'text-embedding-3-large' for OpenAI, custom name for vLLM)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API key for authentication (if not provided, uses environment variable)")
    parser.add_argument("--api_base", type=str, default=None,
                       help="API base URL (None for OpenAI, custom URL for vLLM, e.g., 'http://localhost:8000')")
    parser.add_argument("--api_max_retries", type=int, default=3,
                       help="Maximum number of retries for failed API requests")
    parser.add_argument("--api_timeout", type=int, default=60,
                       help="Request timeout in seconds for API calls")
    parser.add_argument("--api_max_tokens_per_text", type=int, default=30000,
                       help="Maximum tokens per individual text for API calls")
    
    # Backward compatibility for OpenAI-specific parameters
    parser.add_argument("--openai_model", type=str, default=None,
                       help="OpenAI model name (alias for --api_model)")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key (alias for --api_key)")
    parser.add_argument("--vllm_model", type=str, default=None,
                       help="vLLM model name (alias for --api_model)")
    parser.add_argument("--vllm_api_base", type=str, default="http://localhost:8000",
                       help="vLLM API base URL (alias for --api_base)")
    
    # Generated profile arguments
    parser.add_argument("--use_generated_profiles", action="store_true",
                       help="Use pre-generated profile files from the generated_profile directory")
    parser.add_argument("--generated_profile_dir", type=str, default=None,
                       help="Directory containing pre-generated profile files (LaMP, LongLaMP, PRISM, etc.)")
    
    args = parser.parse_args()
    
    # Handle backward compatibility and auto-detection of API type
    if args.openai_model or args.openai_api_key:
        # Using OpenAI API
        args.use_api = True
        args.api_model = args.api_model or args.openai_model or "text-embedding-3-large"
        args.api_key = args.api_key or args.openai_api_key
        args.api_base = args.api_base or None  # Use default OpenAI API
    elif args.vllm_model or (args.vllm_api_base != "http://localhost:8000"):
        # Using vLLM API
        args.use_api = True
        args.api_model = args.api_model or args.vllm_model
        args.api_base = args.api_base or args.vllm_api_base
    elif args.use_api and not args.api_model:
        # Default to OpenAI if API mode requested but no specific model
        args.api_model = "text-embedding-3-large"
    
    # Handle multi-GPU flag
    use_multi_gpu = args.use_multi_gpu and not args.no_multi_gpu
    
    # Log configuration
    logger.info("=== Configuration ===")
    if args.use_api:
        logger.info(f"Embedding mode: API")
        logger.info(f"API model: {args.api_model}")
        if args.api_base:
            logger.info(f"API base: {args.api_base}")
        else:
            logger.info("API: OpenAI (default)")
        logger.info(f"API batch size: {args.batch_size}")
        logger.info(f"API max retries: {args.api_max_retries}")
        logger.info(f"API timeout: {args.api_timeout}s")
        logger.info(f"API max tokens per text: {args.api_max_tokens_per_text}")
    else:
        logger.info(f"Embedding mode: Local model")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Base batch size: {args.batch_size}")
        logger.info(f"Multi-GPU: {use_multi_gpu}")
        logger.info(f"Max memory per GPU: {args.max_memory_gb}GB")
    logger.info(f"Num workers: {args.num_workers}")
    logger.info(f"Force regenerate: {args.force_regenerate}")
    logger.info(f"Use generated profiles: {args.use_generated_profiles}")
    if args.use_generated_profiles:
        logger.info(f"Generated profile dir: {args.generated_profile_dir}")
    
    # Validation for generated profiles
    if args.use_generated_profiles and not args.generated_profile_dir:
        logger.error("Error: --use_generated_profiles requires --generated_profile_dir to be specified")
        return
    
    if args.use_generated_profiles and not os.path.exists(args.generated_profile_dir):
        logger.error(f"Error: Generated profile directory does not exist: {args.generated_profile_dir}")
        return
    
    # Handle cleanup of temporary files if requested
    if args.cleanup_temp_files:
        logger.info("=== Cleanup Mode ===")
        cleanup_temporary_files(args.output_dir)
        logger.info("Cleanup completed, exiting.")
        return
    
    # Initialize the generator
    generator = UserEmbeddingGenerator(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        use_multi_gpu=use_multi_gpu,
        max_memory_gb=args.max_memory_gb,
        num_workers=args.num_workers,
        force_regenerate=args.force_regenerate,
        # API parameters
        use_api=args.use_api,
        api_key=args.api_key,
        api_base=args.api_base,
        api_model=args.api_model,
        api_max_retries=args.api_max_retries,
        api_timeout=args.api_timeout,
        api_max_tokens_per_text=args.api_max_tokens_per_text,
        # Generated profile parameters
        use_generated_profiles=args.use_generated_profiles,
        generated_profile_dir=args.generated_profile_dir
    )
    
    # Generate embeddings
    if args.dataset and args.task:
        # Process specific dataset and task
        dataset_path = os.path.join(args.data_dir, args.dataset)
        if os.path.exists(dataset_path):
            result = generator.generate_embeddings_for_dataset(dataset_path, args.dataset, args.output_dir, args.task)
            if result:
                logger.info(f"Completed processing {args.dataset}_{args.task}: {result['num_users']} users in {result['processing_time']:.2f}s")
        else:
            logger.error(f"Dataset directory not found: {dataset_path}")
    elif args.dataset:
        # Process specific dataset
        dataset_path = os.path.join(args.data_dir, args.dataset)
        if os.path.exists(dataset_path):
            result = generator.generate_embeddings_for_dataset(dataset_path, args.dataset, args.output_dir)
            if result:
                logger.info(f"Completed processing {args.dataset}: {result['num_users']} users in {result['processing_time']:.2f}s")
        else:
            logger.error(f"Dataset directory not found: {dataset_path}")
    else:
        # Generate embeddings for all datasets
        generator.generate_all_embeddings(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main() 
