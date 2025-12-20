import argparse
import json
import logging
import os
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import numpy as np
import time

# Set environment variables to disable HTTP requests to HuggingFace Hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"

import warnings
# Suppress specific warnings about fetching remote files
warnings.filterwarnings("ignore", message="Unable to fetch remote file.*", module="peft")
warnings.filterwarnings("ignore", message="Could not find a config file.*", module="peft")

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, IA3Config, PromptTuningConfig, get_peft_model, PeftModel
from rank_bm25 import BM25Okapi

from hyper_llm_modulator.configs import ArgumentParser
from hyper_llm_modulator.utils.eval_tasks import (
    LaMPClassificationTask, 
    LaMPRatingTask, 
    LaMPTextGenerationTask,
    ClassificationSample,
    RatingPredictionSample,
    TextGenerationSample
)
from hyper_llm_modulator.utils.preprocessing import get_preprocessing_fn
from fishfarm.models import Message

logger = logging.getLogger(__name__)


class OPPUTrainer:
    """
    On-the-fly Personalized PEFT Training for User-specific Adaptation
    Adapted from the original OPPU.py script for text-to-lora codebase
    """
    
    def __init__(
        self,
        model_name: str,
        data_path: str,
        dataset_name: str,
        output_dir: str,
        max_length: int = 2048,
        num_retrieved_examples: int = 0,
        batch_size: int = 8,  # Match baseline
        learning_rate: float = 8e-5,  # Match baseline
        num_epochs: int = 100,  # Match baseline
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,  # Match baseline
        peft_method: str = "lora",
        gradient_accumulation_steps: int = 4,  # Default for better memory usage
        gradient_checkpointing: bool = True,  # Enable gradient checkpointing to reduce memory
        access_token: Optional[str] = None,
        use_checkpoints: bool = True,  # Whether to save/load checkpoints
        data_version: str = "auto",  # Version of the data to use (auto, v2, v3, v4, etc.)
    ):
        self.model_name = model_name
        self.data_path = Path(data_path)
        self.full_dataset_name = dataset_name  # Store full name with test type for file naming
        
        # Parse dataset name to extract base name and test type
        if dataset_name.endswith('_random_test'):
            self.dataset_name = dataset_name[:-12]  # Remove '_random_test'
            self.test_type = 'random_test'
        elif dataset_name.endswith('_ood_test'):
            self.dataset_name = dataset_name[:-9]   # Remove '_ood_test'
            self.test_type = 'ood_test'
        else:
            # Fallback: assume random_test if no suffix specified
            self.dataset_name = dataset_name
            self.test_type = 'random_test'
            self.full_dataset_name = f"{dataset_name}_random_test"
            
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.num_retrieved_examples = num_retrieved_examples
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.peft_method = self._normalize_peft_method(peft_method)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.access_token = access_token
        self.use_checkpoints = use_checkpoints
        self.data_version = data_version
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory if checkpoints are enabled
        if self.use_checkpoints:
            (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        
        # Load prompt template from metadata.yaml
        self.user_prompt_template = self._load_prompt_template()
        
        # Initialize model and tokenizer
        self._setup_model_and_tokenizer()
        
        # Resolve actual data version if auto was specified
        self.actual_data_version = self._resolve_data_version()
        
        # Load dataset
        self.test_data = self._load_test_data()
        
        # Initialize preprocessing function for this dataset
        self.preprocessing_fn = self._get_preprocessing_fn()

        logger.info(f"Using PEFT method: {self.peft_method}")

        # Cache for reusable module metadata
        self._cached_linear_modules: Optional[set] = None

    def _normalize_peft_method(self, method: Optional[str]) -> str:
        """Normalize and validate the requested PEFT method"""
        if not method:
            return "lora"

        method_map = {
            "lora": "lora",
            "ia3": "ia3",
            "ia-3": "ia3",
            "prompt_tuning": "prompt_tuning",
            "prompt-tuning": "prompt_tuning",
            "prompt": "prompt_tuning",
        }

        normalized = method_map.get(method.lower())
        if not normalized:
            valid = ["lora", "ia3", "prompt_tuning"]
            raise ValueError(f"Unsupported PEFT method '{method}'. Valid options: {valid}")

        return normalized
        
    def _parse_version_number(self, version_str: str) -> int:
        """Parse version string (e.g., 'v3', 'v4') to get version number"""
        if version_str.startswith('v') and version_str[1:].isdigit():
            return int(version_str[1:])
        return 0
    
    def _find_available_versions(self, data_dir: Path) -> List[str]:
        """Find all available version directories in data_dir"""
        available_versions = []
        if not data_dir.exists():
            return available_versions
            
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.startswith('v') and item.name[1:].isdigit():
                available_versions.append(item.name)
        # Sort by version number, highest first
        available_versions.sort(key=self._parse_version_number, reverse=True)
        return available_versions
    
    def _resolve_data_version(self) -> str:
        """Resolve the actual data version to use, handling 'auto' detection"""
        if self.data_version == 'auto':
            # Find highest available version
            if self.dataset_name.startswith("LaMP"):
                data_dir = self.data_path / "LaMP"
            elif self.dataset_name.startswith("LongLaMP"):
                data_dir = self.data_path / "LongLaMP"
            else:
                logger.warning(f"Unknown dataset type for auto version detection: {self.dataset_name}")
                return "v2"  # fallback
                
            available_versions = self._find_available_versions(data_dir)
            if available_versions:
                chosen_version = available_versions[0]
                logger.info(f"Auto-detected data version: {chosen_version} for dataset {self.dataset_name}")
                return chosen_version
            else:
                logger.warning(f"No version directories found in {data_dir}, using v2 as fallback")
                return "v2"
        else:
            # Validate specified version exists
            if self.dataset_name.startswith("LaMP"):
                data_dir = self.data_path / "LaMP"
            elif self.dataset_name.startswith("LongLaMP"):
                data_dir = self.data_path / "LongLaMP"
            else:
                return self.data_version  # trust user input for unknown datasets
                
            version_dir = data_dir / self.data_version
            if not version_dir.exists():
                logger.warning(f"Specified version {self.data_version} not found in {data_dir}, using as-is")
            return self.data_version
        
    def _load_prompt_template(self) -> str:
        """Load prompt template from metadata.yaml file"""
        # Determine the task directory name based on dataset name
        if self.dataset_name.startswith("LaMP"):
            # Convert LaMP_processed_movie to lamp_movie_random_test or lamp_movie_ood_test
            task_name = self.dataset_name.lower().replace("lamp_processed_", "lamp_") + f"_{self.test_type}"
        elif self.dataset_name.startswith("LongLaMP"):
            # Convert LongLaMP_product_review to longlamp_product_review_random_test or longlamp_product_review_ood_test
            task_name = self.dataset_name.lower().replace("longlamp_", "longlamp_") + f"_{self.test_type}"
        else:
            # Fallback for other dataset names
            task_name = self.dataset_name.lower() + f"_{self.test_type}"
        
        # Construct metadata.yaml path
        tasks_dir = Path("./tasks")
        metadata_path = tasks_dir / task_name / "metadata.yaml"
        
        # Try alternative task directory names if the first doesn't exist
        if not metadata_path.exists():
            # Try with all_history_ prefix
            alt_task_name = f"all_history_{task_name}"
            metadata_path = tasks_dir / alt_task_name / "metadata.yaml"
            
        if not metadata_path.exists():
            # Try with RAG_ prefix
            alt_task_name = f"RAG_{task_name}"
            metadata_path = tasks_dir / alt_task_name / "metadata.yaml"
            
        if not metadata_path.exists():
            logger.warning(f"Could not find metadata.yaml for dataset {self.dataset_name}")
            logger.warning(f"Tried paths: {task_name}, all_history_{task_name}, RAG_{task_name}")
            return None
            
        # Load and parse the metadata.yaml file
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            template = metadata.get('user_prompt_template', '')
            logger.info(f"Loaded prompt template from: {metadata_path}")
            logger.info(f"Template: {template[:100]}...")
            return template
            
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_path}: {e}")
            return None
        
    def _get_preprocessing_fn(self):
        """Get the appropriate preprocessing function for this dataset"""
        dataset_lower = self.dataset_name.lower()
        
        # Map dataset names to preprocessing function names
        if "movie" in dataset_lower:
            ds_name = "lamp_movie"
        elif "citation" in dataset_lower:
            ds_name = "lamp_citation"  
        elif "product" in dataset_lower and "lamp" in dataset_lower and "longlamp" not in dataset_lower:
            ds_name = "lamp_product"
        elif "tweet" in dataset_lower:
            ds_name = "lamp_tweet"
        elif "news_headline" in dataset_lower:
            ds_name = "lamp_news_headline"
        elif "news_cat" in dataset_lower:
            ds_name = "lamp_news_cat"
        elif "scholarly" in dataset_lower:
            ds_name = "lamp_scholarly_title"
        elif "product_review" in dataset_lower and "long" in dataset_lower:
            ds_name = "longlamp_product_review"
        elif "abstract_generation" in dataset_lower and "long" in dataset_lower:
            ds_name = "longlamp_abstract_generation"
        elif "topic_writing" in dataset_lower and "long" in dataset_lower:
            ds_name = "longlamp_topic_writing"
        else:
            # Default identity function for unknown datasets
            ds_name = "unknown"
            
        return get_preprocessing_fn(ds_name)
        
    def _parse_query_input(self, query: Dict) -> Dict:
        """Parse query input using the preprocessing function to extract structured fields"""
        # Create a mock example with the query data for preprocessing
        mock_example = {
            "input": query.get("input", ""),
            "output": query.get("gold", "") or query.get("output", ""),  # Use gold as output for preprocessing, fallback to output
            "profile_text": "",  # These won't be used for parsing input structure
            "profile_retrieval_k1": [],
            "profile_retrieval_k2": [],
            "profile_retrieval_k4": [],
            "profile_all_history": ""  # Add missing profile_all_history field
        }
        
        try:
            # Apply preprocessing to extract structured fields
            parsed = self.preprocessing_fn(mock_example)
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse query input with preprocessing function: {e}")
            # Fallback to original input
            return {"input": query.get("input", "")}
    
    def _format_query_with_template(self, parsed_query: Dict) -> str:
        """Format the parsed query using the loaded prompt template"""
        if not self.user_prompt_template:
            # Fallback to using input directly if no template
            return parsed_query.get("input", "")
            
        try:
            # Format the template with parsed fields
            formatted_prompt = self.user_prompt_template.format(**parsed_query)
            return formatted_prompt
        except KeyError as e:
            logger.warning(f"Missing template variable {e} in parsed query")
            # Fallback to using input directly
            return parsed_query.get("input", "")
        except Exception as e:
            logger.warning(f"Error formatting template with parsed query: {e}")
            return parsed_query.get("input", "")
        
    def _setup_model_and_tokenizer(self):
        """Initialize the base model and tokenizer"""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            token=self.access_token,
            local_files_only=False  # Keep False for initial loading, only set offline for training
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            token=self.access_token,
            local_files_only=False  # Keep False for initial loading, only set offline for training
        )
        
        # Disable cache for training; required for gradient checkpointing
        self.base_model.config.use_cache = False
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Optionally enable gradient checkpointing on the base model
        if self.gradient_checkpointing and hasattr(self.base_model, "gradient_checkpointing_enable"):
            try:
                self.base_model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing on base model")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing on base model: {e}")
        if self.gradient_checkpointing and hasattr(self.base_model, "enable_input_require_grads"):
            try:
                self.base_model.enable_input_require_grads()
            except Exception as e:
                logger.debug(f"Could not set input require grads on base model: {e}")
        
    def _load_test_data(self) -> List[Dict]:
        """Load test data from specified test split"""
        if self.dataset_name.startswith("LaMP"):
            filename = f"{self.dataset_name}_{self.test_type}.jsonl"
            filepath = self.data_path / "LaMP" / self.actual_data_version / filename
        elif self.dataset_name.startswith("LongLaMP"):
            filename = f"{self.dataset_name}_{self.test_type}.jsonl"
            filepath = self.data_path / "LongLaMP" / self.actual_data_version / filename
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        logger.info(f"Loading test data from: {filepath}")
        logger.info(f"Using data version: {self.actual_data_version}")
        
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
                
        logger.info(f"Loaded {len(data)} test users from {self.test_type} split")
        return data
        
    def _create_bm25_retriever(self, history: List[Dict]) -> Optional[BM25Okapi]:
        """Create BM25 retriever from user history"""
        if not history or self.num_retrieved_examples <= 0:
            return None
            
        # Extract text content for BM25 indexing
        documents = []
        for item in history:
            # Combine available text fields
            text_parts = []
            if 'input' in item:
                text_parts.append(item['input'])
            if 'output' in item:
                text_parts.append(item['output'])
            if 'reviewText' in item:
                text_parts.append(item['reviewText'])
            if 'summary' in item:
                text_parts.append(item['summary'])
            if 'description' in item:
                text_parts.append(item['description'])
                
            documents.append(' '.join(text_parts))
            
        if not documents:
            return None
            
        tokenized_corpus = [doc.split() for doc in documents]
        return BM25Okapi(tokenized_corpus)
        
    def _retrieve_relevant_history(
        self, 
        query: Dict, 
        history: List[Dict], 
        bm25: Optional[BM25Okapi]
    ) -> List[Dict]:
        """Retrieve relevant history items using BM25"""
        if not bm25 or not history:
            return history[:self.num_retrieved_examples]
            
        # Extract query text
        query_parts = []
        if 'input' in query:
            query_parts.append(query['input'])
        if 'output' in query:
            query_parts.append(query['output'])
            
        query_text = ' '.join(query_parts)
        tokenized_query = query_text.split()
        
        # Get top-k most relevant documents
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-self.num_retrieved_examples:][::-1]
        
        return [history[i] for i in top_indices if i < len(history)]
        
    def _format_prompt_template(self, item: Dict) -> Optional[str]:
        """Format the prompt template with available data fields"""
        if not self.user_prompt_template:
            return None
            
        # Create a mapping of common field names to template placeholders
        template_vars = {}
        
        # Dataset-specific field mappings to fix template variable mismatches
        if "product" in self.dataset_name.lower():
            if self.dataset_name.lower().startswith("lamp"):
                # LaMP Product
                field_mappings = {
                    'review_text': ['text'],
                    'output': ['score'],
                }
            else:
                field_mappings = {
                    'product_description': ['description'],
                    'rating': ['overall'],
                    'review_summary': ['summary'],
                    'output': ['reviewText'],
                }
        elif "tweet" in self.dataset_name.lower():
            # LaMP Tweet: text -> original_tweet
            field_mappings = {
                'original_tweet': ['text'],
            }
        elif "citation" in self.dataset_name.lower():
            # LaMP Citation: title -> paper_title
            field_mappings = {
                'paper_title': ['title'],
                'citation': ['citation'],
            }
        elif "abstract" in self.dataset_name.lower() and "long" in self.dataset_name.lower():
            # LongLaMP Abstract Generation: title -> paper_title
            field_mappings = {
                'paper_title': ['title'],
                # 'key_items': ['key_items', 'items'],  # May not exist in history
                'output': ['abstract'],
            }
        elif "topic" in self.dataset_name.lower():
            # LongLaMP Topic Writing
            field_mappings = {
                'topic_prompt': ['summary'],
                'output': ['content'],
            }
        elif "movie" in self.dataset_name.lower():
            # LaMP Movie
            field_mappings = {
                'description': ['description'],
                'output': ['tag'],
            }
        elif "scholarly" in self.dataset_name.lower():
            # LaMP Scholarly Title
            field_mappings = {
                'abstract': ['abstract'],
                'output': ['title'],
            }
        elif "news_headline" in self.dataset_name.lower():
            # LaMP News (headline or category)
            field_mappings = {
                'article': ['text', 'content', 'article'],
                'output': ['title'],
            }
        elif "news_cat" in self.dataset_name.lower():
            # LaMP News Category
            field_mappings = {
                'article': ['text', 'content', 'article'],
                'output': ['category'],
            }
        
        
        
        # Populate template variables from item data
        for template_var, possible_fields in field_mappings.items():
            for field in possible_fields:
                if field in item and item[field]:
                    template_vars[template_var] = item[field]
                    break
        
        # print("-"*100)
        # print(template_vars)
        # print(self.user_prompt_template)
        try:
            # Format the template with available variables
            formatted_prompt = self.user_prompt_template.format(**template_vars)
            return formatted_prompt, template_vars
        except KeyError as e:
            # Missing required template variable
            logger.debug(f"Missing template variable {e} for item: {list(item.keys())}")
            return None
        except Exception as e:
            logger.warning(f"Error formatting template: {e}")
            return None
        
    def _prepare_training_data(self, history: List[Dict]) -> List[Dict]:
        """Prepare training data from user history"""
        training_examples = []
        
        # Check if this is a citation or tweet task that needs unsupervised training
        is_citation_task = "citation" in self.dataset_name.lower()
        is_tweet_task = "tweet" in self.dataset_name.lower()
        
        for item in history:
            # Handle citation and tweet tasks with unsupervised causal LLM training
            if is_citation_task:
                # Format: "paper title: {title} reference: {citation}"
                if 'title' in item and 'citation' in item:
                    text = f"# Paper Title: {item['title']} # Reference: {item['citation']}"
                    training_examples.append({
                        "text": text,
                        "prompt": None,  # No prompt for unsupervised training
                        "response": None  # No response for unsupervised training
                    })
                continue
                
            elif is_tweet_task:
                # Format: "tweet: {text}"
                if 'text' in item:
                    text = f"# Tweet: {item['text']}"
                    training_examples.append({
                        "text": text,
                        "prompt": None,  # No prompt for unsupervised training
                        "response": None  # No response for unsupervised training
                    })
                continue
            
            # For all other tasks, use supervised training format
            prompt = None
            response = None
            
            # Try to use the loaded prompt template first
            if self.user_prompt_template:
                template_result = self._format_prompt_template(item)
                if template_result is not None:
                    prompt, template_vars = template_result
                    response = template_vars.get('output')
                else:
                    prompt = None
                    response = None
            # print("-"*100)
            # print(prompt)
            # print(response)
            # print("-"*100)

                
            # Ensure we have valid prompt and response
            if not prompt or not response or not prompt.strip() or not response.strip():
                continue
                
            # Create conversation format for supervised tasks
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            full_text = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Truncate if the training example is too long
            if len(self.tokenizer.encode(full_text)) > self.max_length:
                full_text = self._truncate_input_intelligently(
                    full_text, 
                    self.max_length - 50,  # Leave some buffer
                    preserve_suffix=True  # Preserve the response part
                )
            
            training_examples.append({
                "text": full_text,
                "prompt": prompt,
                "response": response
            })
            
        if is_citation_task or is_tweet_task:
            task_type = "citation" if is_citation_task else "tweet"
            logger.info(f"Prepared {len(training_examples)} unsupervised {task_type} training examples")
        else:
            logger.info(f"Prepared {len(training_examples)} supervised training examples using chat templates")
        return training_examples
        
    def _tokenize_data(self, examples: List[Dict]) -> Dict:
        """Tokenize training examples"""
        texts = [ex["text"] for ex in examples]
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,  # Enable padding to ensure consistent lengths
            return_tensors=None,
        )
        
        # For unsupervised training (citation/tweet tasks), labels are just input_ids
        # For supervised training, labels are also input_ids (causal LM training)
        # Ensure labels have the same structure as input_ids
        if isinstance(tokenized["input_ids"], list):
            tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        else:
            tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
        
    def _truncate_input_intelligently(self, text: str, max_length: int, preserve_suffix: bool = True) -> str:
        """Intelligently truncate input text while preserving important parts"""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= max_length:
            return text
            
        if preserve_suffix:
            # Truncate from the left to preserve the end (main query)
            truncated_tokens = tokens[-max_length:]
        else:
            # Truncate from the right
            truncated_tokens = tokens[:max_length]
            
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        logger.debug(f"Truncated input from {len(tokens)} to {len(truncated_tokens)} tokens")
        return truncated_text

    def _clean_chat_template_artifacts(self, text: str) -> str:
        """Clean up chat template artifacts from generated text"""
        # Common chat template patterns to remove
        patterns_to_remove = [
            "<|im_start|>assistant\n",
            "<|im_start|>assistant",
            "<|im_end|>", 
            "<|im_start|>",
            "[INST]",
            "[/INST]",
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            "assistant\n",
            "assistant",
            "system\n",
            "user\n",
        ]
        
        cleaned = text.strip()
        
        # Remove starting patterns
        for pattern in patterns_to_remove:
            if cleaned.startswith(pattern):
                cleaned = cleaned[len(pattern):].strip()
                
        # Remove ending patterns  
        for pattern in patterns_to_remove:
            if cleaned.endswith(pattern):
                cleaned = cleaned[:-len(pattern)].strip()
                
        # Remove any remaining template tokens
        for pattern in patterns_to_remove:
            cleaned = cleaned.replace(pattern, "").strip()
        
        # Additional cleanup for Qwen-specific artifacts
        if cleaned.startswith("You are Qwen"):
            # Find the actual response after system message
            lines = cleaned.split('\n')
            # Look for the last line that contains actual content
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line and not any(pattern in line.lower() for pattern in ["system", "user", "assistant", "qwen", "alibaba"]):
                    cleaned = line
                    break
        
        return cleaned
        
    def _ensure_clean_base_model(self):
        """Ensure the base model is in a clean state without any PEFT adapters"""
        if isinstance(self.base_model, PeftModel):
            logger.debug("Unloading existing PEFT adapters from base model")
            self.base_model = self.base_model.unload()
        
        # Force clean PEFT state by checking for any remaining peft configs
        if hasattr(self.base_model, 'peft_config') and self.base_model.peft_config:
            logger.warning("Base model still has peft_config, attempting to clean")
            # Remove peft config attributes
            if hasattr(self.base_model, 'peft_config'):
                del self.base_model.peft_config
            if hasattr(self.base_model, 'peft_type'):
                del self.base_model.peft_type
                
        logger.debug(f"Base model state: {type(self.base_model).__name__}")

    def _get_linear_module_names(self) -> List[str]:
        """Collect names of modules with weights for PEFT targeting"""
        if self._cached_linear_modules is not None:
            return sorted(self._cached_linear_modules)

        modules = set()
        for name, module in self.base_model.named_modules():
            if hasattr(module, 'weight'):
                modules.add(name.split('.')[-1])

        self._cached_linear_modules = modules
        logger.debug(f"Discovered linear modules: {sorted(modules)}")
        return sorted(modules)

    def _resolve_module_names(
        self,
        preferred: Optional[List[str]] = None,
        fallback_keywords: Optional[List[str]] = None,
        default: Optional[List[str]] = None,
    ) -> List[str]:
        """Resolve module names prioritising preferred targets and keyword fallbacks"""
        available = self._get_linear_module_names()

        if preferred:
            matched = [module for module in preferred if module in available]
            if matched:
                logger.debug(f"Resolved preferred modules: {matched}")
                return matched

        if fallback_keywords:
            keyword_matches = [
                module
                for module in available
                if any(keyword in module for keyword in fallback_keywords)
            ]
            if keyword_matches:
                logger.debug(f"Resolved keyword-matched modules: {keyword_matches}")
                return keyword_matches

        if available:
            logger.debug(f"Falling back to all available modules: {available}")
            return available

        default = default or preferred or []
        logger.debug(f"No modules discovered; using default fallback: {default}")
        return default

    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration"""
        target_modules = self._resolve_module_names(
            preferred=["q_proj", "v_proj"],
            fallback_keywords=["proj", "projection"],
            default=["q_proj", "v_proj"],
        )

        logger.info(f"LoRA target modules: {target_modules}")

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,  # Match baseline
        )

    def _create_ia3_config(self) -> IA3Config:
        """Create IA3 configuration"""
        attention_modules = self._resolve_module_names(
            preferred=["q_proj", "k_proj", "v_proj"],
            fallback_keywords=["proj"],
        )
        feedforward_modules = self._resolve_module_names(
            preferred=["up_proj", "down_proj", "gate_proj"],
            fallback_keywords=["fc", "ff", "mlp", "proj"],
        )

        # Ensure target modules include every feedforward module to satisfy PEFT validation
        target_modules = []
        seen = set()
        for name in attention_modules + feedforward_modules:
            if name not in seen:
                target_modules.append(name)
                seen.add(name)
        if not target_modules:
            target_modules = self._resolve_module_names(default=["q_proj", "k_proj", "v_proj"])

        logger.info(
            "IA3 target modules: %s; feedforward modules: %s",
            target_modules,
            feedforward_modules,
        )

        return IA3Config(
            target_modules=target_modules,
            feedforward_modules=[name for name in feedforward_modules if name in target_modules],
            task_type="CAUSAL_LM",
        )

    def _create_prompt_tuning_config(self) -> PromptTuningConfig:
        """Create Prompt Tuning configuration"""
        hidden_size = getattr(self.base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.base_model.config, "n_embd", None)

        if hidden_size is None:
            raise ValueError("Could not determine model hidden size for prompt tuning")

        num_virtual_tokens = max(int(self.lora_r), 8)
        logger.info(
            f"Prompt tuning virtual tokens: {num_virtual_tokens}; hidden size: {hidden_size}"
        )

        return PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init="RANDOM",
            num_virtual_tokens=num_virtual_tokens,
            token_dim=hidden_size,
        )

    def _create_peft_config(self):
        """Create the PEFT config for the configured method"""
        if self.peft_method == "lora":
            return self._create_lora_config()
        if self.peft_method == "ia3":
            return self._create_ia3_config()
        if self.peft_method == "prompt_tuning":
            return self._create_prompt_tuning_config()
        raise ValueError(f"Unsupported PEFT method: {self.peft_method}")

    def _config_from_dict(
        self,
        config_dict: Dict,
        method: Optional[str],
    ) -> Union[LoraConfig, IA3Config, PromptTuningConfig]:
        """Reconstruct a PEFT config object from a serialized dictionary"""
        if not isinstance(config_dict, dict):
            raise TypeError("config_dict must be a dictionary")

        normalized_method = self._normalize_peft_method(method or self.peft_method)

        config_map = {
            "lora": LoraConfig,
            "ia3": IA3Config,
            "prompt_tuning": PromptTuningConfig,
        }

        config_cls = config_map[normalized_method]
        valid_fields = getattr(config_cls, '__dataclass_fields__', {}).keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}

        if normalized_method == "prompt_tuning":
            hidden_size = getattr(self.base_model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.base_model.config, "n_embd", None)
            if 'token_dim' not in filtered and hidden_size is not None:
                filtered['token_dim'] = hidden_size
            if 'num_virtual_tokens' not in filtered:
                filtered['num_virtual_tokens'] = max(int(self.lora_r), 8)

        if 'task_type' not in filtered:
            filtered['task_type'] = "CAUSAL_LM"

        return config_cls(**filtered)
        
    def _train_user_adapter(self, user_id: str, training_data: List[Dict]) -> Optional[Dict]:
        """Train a PEFT adapter for a specific user and return weights in memory"""
        if not training_data:
            logger.warning(f"No training data for user {user_id}")
            return None
            
        logger.info(
            f"Training {self.peft_method} adapter for user {user_id} with {len(training_data)} examples"
        )
        
        # Ensure base model is clean before training
        self._ensure_clean_base_model()
        
        # Ensure base model is in training mode initially
        self.base_model.train()
        
        # Create fresh model instance for each user
        peft_config = self._create_peft_config()
        model = get_peft_model(self.base_model, peft_config)
        
        # Verify that adapters were actually added
        if not hasattr(model, 'peft_config') or not model.peft_config:
            raise RuntimeError(
                f"Failed to create PEFT model with adapters for user {user_id}"
            )
            
        logger.info(f"Created PEFT model with config: {peft_config}")
        
        # Configure model for training
        model.train()

        # Enable gradient checkpointing on PEFT-wrapped model if requested
        if self.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing on PEFT model")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing on PEFT model: {e}")
        if self.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception as e:
                logger.debug(f"Could not set input require grads on PEFT model: {e}")
        
        # Disable cache for training (important for gradient computation)
        if hasattr(model, 'config'):
            model.config.use_cache = False
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
            model.base_model.config.use_cache = False
        
        # Ensure adapter parameters are trainable and base model parameters are frozen
        trainable_param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
            else:
                param.requires_grad_(False)

        logger.info(
            f"Found {len(trainable_param_names)} trainable parameters for {self.peft_method}: {trainable_param_names[:5]}"
            + ("..." if len(trainable_param_names) > 5 else "")
        )
        
        # Verify that some parameters require gradients
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        if trainable_params == 0:
            raise RuntimeError(f"No trainable parameters found for user {user_id}! This will cause training to fail.")
            
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # Prepare dataset
        tokenized_data = self._tokenize_data(training_data)
        dataset = Dataset.from_dict(tokenized_data)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"temp_{user_id}"),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,  # Use gradient_accumulation_steps
            gradient_checkpointing=self.gradient_checkpointing,
            optim='adamw_torch',
            num_train_epochs=self.num_epochs,
            save_steps=1e9,  # Don't save during training
            logging_steps=50,
            learning_rate=self.learning_rate,
            weight_decay=1e-2,
            bf16=True,
            max_grad_norm=1.0,  # Match baseline
            warmup_ratio=0.1,  # Match baseline
            lr_scheduler_type='linear',
            report_to='none',
            remove_unused_columns=False,
            hub_token=None,  # Disable hub integration
            push_to_hub=False,  # Disable hub integration
        )
        
        # Create data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # This is for causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficiency
            return_tensors="pt"
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Extract adapter weights in memory instead of saving to disk
        adapter_state_dict = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Also get the adapter config for reconstruction
        adapter_config = model.peft_config['default']
        
        # Cleanup GPU memory immediately
        del model
        del trainer
        del dataset
        del data_collator
        torch.cuda.empty_cache()
        
        # Ensure base model is restored to clean state
        self._ensure_clean_base_model()
        
        # Cleanup temp directory
        import shutil
        temp_dir = self.output_dir / f"temp_{user_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        logger.info(
            f"Successfully trained {self.peft_method} adapter for user {user_id}, "
            f"extracted {len(adapter_state_dict)} parameters"
        )
        
        return {
            'adapter_weights': adapter_state_dict,
            'peft_config': adapter_config,
            'peft_method': self.peft_method,
            'user_id': user_id
        }
        
    def _evaluate_user(self, user_data: Dict, adapter_data: Optional[Dict]) -> List[Dict]:
        """Evaluate user queries with a trained adapter from memory"""
        user_id = user_data['user_id']
        queries = user_data['query']
        history = user_data.get('history', [])
        
        logger.info(f"Evaluating user {user_id} with {len(queries)} queries using chat templates")
        
        model = self.base_model
        used_adapter = False
        adapter_method = None

        # Load model with adapter if available
        if adapter_data and adapter_data.get('adapter_weights'):
            adapter_method = adapter_data.get('peft_method')
            if adapter_method is None and 'lora_config' in adapter_data:
                adapter_method = 'lora'
            adapter_method = adapter_method or self.peft_method

            logger.info(
                f"Loading {adapter_method} adapter for user {user_id} with "
                f"{len(adapter_data['adapter_weights'])} parameters"
            )

            # Ensure base model is clean before creating PEFT model
            self._ensure_clean_base_model()

            try:
                adapter_config = adapter_data.get('peft_config') or adapter_data.get('lora_config')
                if adapter_config is None:
                    raise ValueError("Adapter config missing from adapter data")

                # If config was serialized as dict, reconstruct it
                if isinstance(adapter_config, dict):
                    adapter_config = self._config_from_dict(adapter_config, adapter_method)

                model = get_peft_model(self.base_model, adapter_config)

                # Load the adapter weights from memory
                adapter_weights = adapter_data['adapter_weights']
                missing_keys, unexpected_keys = model.load_state_dict(adapter_weights, strict=False)

                if missing_keys:
                    logger.warning(
                        f"Missing keys when loading {adapter_method} adapter for user {user_id}: {missing_keys}"
                    )
                if unexpected_keys:
                    logger.warning(
                        f"Unexpected keys when loading {adapter_method} adapter for user {user_id}: {unexpected_keys}"
                    )

                used_adapter = True
                logger.info(f"Successfully loaded {adapter_method} adapter for user {user_id}")

            except Exception as e:
                logger.error(f"Error loading adapter for user {user_id}: {e}")
                logger.info(f"Falling back to base model for user {user_id}")
                model = self.base_model
                used_adapter = False
                self._ensure_clean_base_model()
        else:
            logger.info(f"No adapter data available for user {user_id}, using base model")
        
        model.eval()
        model.config.use_cache = True
        
        # Setup BM25 for retrieval
        bm25 = self._create_bm25_retriever(history)
        
        results = []
        
        for query in tqdm(queries, desc=f"Evaluating {user_id}"):
            # Parse query input using preprocessing function
            parsed_query = self._parse_query_input(query)
            
            # Format the parsed query using the loaded prompt template
            prompt = self._format_query_with_template(parsed_query)
            
            # Retrieve relevant history
            relevant_history = self._retrieve_relevant_history(query, history, bm25)
            
            # Format retrieved context using chat templates for consistency
            context = ""
            if relevant_history:
                context_examples = []
                # Calculate available space for context (reserve space for prompt and generation)
                max_context_length = self.max_length - 400  # More room for prompt and generation
                current_context_length = 0
                
                for item in relevant_history:
                    if 'input' in item and 'output' in item:
                        # Format each example as a proper chat conversation
                        example_conversation = [
                            {"role": "user", "content": item['input']},
                            {"role": "assistant", "content": item['output']}
                        ]
                        example_text = self.tokenizer.apply_chat_template(
                            example_conversation,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        
                        # Check if adding this example would exceed context length
                        example_tokens = len(self.tokenizer.encode(example_text))
                        if current_context_length + example_tokens > max_context_length:
                            logger.debug(f"Truncating context for user {user_id}: reached max context length")
                            break
                            
                        context_examples.append(example_text)
                        current_context_length += example_tokens
                        
                if context_examples:
                    context = "Here are some relevant examples:\n" + "\n".join(context_examples) + "\n\n"
            
            # Create conversation for chat template
            user_message = context + prompt
            conversation = [{"role": "user", "content": user_message}]
            
            # Apply chat template with generation prompt
            full_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Check final prompt length and truncate if necessary
            max_input_length = self.max_length - 300  # Leave room for generation
            
            if len(self.tokenizer.encode(full_prompt)) > max_input_length:
                logger.debug(f"Truncating final prompt for user {user_id}")
                full_prompt = self._truncate_input_intelligently(
                    full_prompt, 
                    max_input_length, 
                    preserve_suffix=True  # Preserve the main query at the end
                )
            
            # Tokenize and generate
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
                padding=False
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2**9,
                    do_sample=True,
                    temperature=1e-7,
                    top_p=1.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            # Decode response and extract assistant's reply
            # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new tokens (assistant response) by removing input tokens
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up any chat template artifacts more robustly
            response = self._clean_chat_template_artifacts(response)
            
            results.append({
                "id": query.get('id', ''),
                "input": query['input'],
                "output": response,
                "gold": query.get('gold', '') or query.get('output', '')
            })
        
        # Clean up GPU memory and restore base model state after evaluation
        if used_adapter:
            adapter_name = adapter_method or self.peft_method
            logger.debug(f"Cleaning up {adapter_name} adapter model for user {user_id}")
            del model
            torch.cuda.empty_cache()

            # Ensure base model is restored to clean state
            self._ensure_clean_base_model()
        
        return results
        
    def _compute_final_metrics(self, all_results: List[Dict]) -> Dict:
        """Compute final evaluation metrics based on task type"""
        logger.info(f"Computing final evaluation metrics for {len(all_results)} predictions")
        
        # Determine task type based on dataset name
        dataset_lower = self.dataset_name.lower()
        
        # Classification tasks: lamp_movie, lamp_citation, lamp_news_cat
        if any(x in dataset_lower for x in ["movie", "citation", "news_cat"]):
            return self._compute_classification_metrics(all_results)
        
        # Rating prediction tasks: lamp_product
        elif "product" in dataset_lower and "lamp" in dataset_lower and "longlamp" not in dataset_lower:
            return self._compute_rating_metrics(all_results)
        
        # Text generation tasks: all others
        else:
            return self._compute_generation_metrics(all_results)
    
    def _compute_classification_metrics(self, all_results: List[Dict]) -> Dict:
        """Compute classification metrics (accuracy, F1)"""
        samples = []
        outputs = []
        
        for result in all_results:
            # Use input as prompt and gold as label
            prompt = result['input']
            label = result['gold']
            output = result['output']
            
            if label:  # Only include samples with valid labels
                samples.append(ClassificationSample(prompt=prompt, label=str(label)))
                outputs.append(output)
        
        if not samples:
            logger.warning("No valid samples for classification evaluation")
            return {"error": "No valid samples"}
        
        # Create evaluator and compute metrics
        evaluator = LaMPClassificationTask(samples=samples, context_messages=[])
        task_result = evaluator.batch_evaluate_with_outputs(outputs)
        
        logger.info(f"Classification metrics: {task_result.aggregate_metrics}")
        return task_result.aggregate_metrics
    
    def _compute_rating_metrics(self, all_results: List[Dict]) -> Dict:
        """Compute rating prediction metrics (MAE, RMSE, STSB)"""
        samples = []
        outputs = []
        
        for result in all_results:
            prompt = result['input']
            rating = result['gold']
            output = result['output']
            
            if rating:  # Only include samples with valid ratings
                try:
                    rating_value = float(rating)
                    samples.append(RatingPredictionSample(prompt=prompt, rating=rating_value))
                    outputs.append(output)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid rating value: {rating}")
                    continue
        
        if not samples:
            logger.warning("No valid samples for rating evaluation")
            return {"error": "No valid samples"}
        
        # Create evaluator and compute metrics
        evaluator = LaMPRatingTask(samples=samples, context_messages=[])
        task_result = evaluator.batch_evaluate_with_outputs(outputs)
        
        logger.info(f"Rating metrics: {task_result.aggregate_metrics}")
        return task_result.aggregate_metrics
    
    def _compute_generation_metrics(self, all_results: List[Dict]) -> Dict:
        """Compute text generation metrics (ROUGE, METEOR)"""
        samples = []
        outputs = []
        
        for result in all_results:
            prompt = result['input']
            reference = result['gold']
            output = result['output']
            
            if reference:  # Only include samples with valid references
                samples.append(TextGenerationSample(prompt=prompt, reference=str(reference)))
                outputs.append(output)
        
        if not samples:
            logger.warning("No valid samples for generation evaluation")
            return {"error": "No valid samples"}
        
        # Create evaluator and compute metrics
        evaluator = LaMPTextGenerationTask(
            samples=samples, 
            context_messages=[],
            rouge_types=("rouge1", "rougeL")  # Standard ROUGE metrics
        )
        task_result = evaluator.batch_evaluate_with_outputs(outputs)
        
        logger.info(f"Generation metrics: {task_result.aggregate_metrics}")
        return task_result.aggregate_metrics
    
    def _evaluate_and_save_user_performance(
        self,
        user_id: str,
        user_results: List[Dict],
        training_time_seconds: float,
        inference_time_seconds: float,
    ) -> Dict:
        """Evaluate and save performance for a single user/task with timing metadata"""
        num_queries = len(user_results)

        if not user_results:
            logger.warning(f"No results for user {user_id}")
            user_metrics = {"error": "No results"}
        else:
            logger.info(f"Evaluating performance for user {user_id} with {num_queries} queries")

            # Compute metrics for this user
            user_metrics = self._compute_final_metrics(user_results)
        
        # Create user-specific performance data
        user_performance = {
            "user_id": user_id,
            "num_queries": num_queries,
            "dataset": self.full_dataset_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_time_seconds": training_time_seconds,
            "inference_time_seconds": inference_time_seconds,
            "metrics": user_metrics,
            "individual_results": user_results
        }
        
        # Save individual user performance file
        user_performance_file = self.output_dir / f"user_{user_id}_performance.json"
        with open(user_performance_file, 'w') as f:
            json.dump(user_performance, f, indent=2)
            
        logger.info(f"User {user_id} performance saved to: {user_performance_file}")
        logger.info(f"User {user_id} metrics: {user_metrics}")

        return {
            "user_id": user_id,
            "metrics": user_metrics,
            "performance_file": str(user_performance_file),
            "num_queries": num_queries,
            "training_time_seconds": training_time_seconds,
            "inference_time_seconds": inference_time_seconds,
        }
    
    def _analyze_user_performance_distribution(self, user_performance_summaries: List[Dict]) -> Dict:
        """Analyze the distribution of user performance metrics"""
        if not user_performance_summaries:
            return {"error": "No user performance data"}
            
        # Extract metric values for analysis
        valid_summaries = [s for s in user_performance_summaries if "error" not in s.get("metrics", {})]
        if not valid_summaries:
            return {"error": "No valid user performance data"}
            
        # Get the first user's metrics to determine what metrics are available
        sample_metrics = valid_summaries[0]["metrics"]
        metric_names = [k for k in sample_metrics.keys() if isinstance(sample_metrics[k], (int, float))]
        
        analysis = {
            "total_users": len(user_performance_summaries),
            "valid_users": len(valid_summaries),
            "metric_distribution": {}
        }
        
        # Analyze distribution for each metric
        for metric_name in metric_names:
            values = []
            for summary in valid_summaries:
                metric_value = summary["metrics"].get(metric_name)
                if isinstance(metric_value, (int, float)):
                    values.append(metric_value)
            
            if values:
                analysis["metric_distribution"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        return analysis
        
    def _get_user_checkpoint_path(self, user_id: str) -> Path:
        """Get the checkpoint path for a specific user"""
        suffix_map = {
            "lora": "lora",
            "ia3": "ia3",
            "prompt_tuning": "prompt",
        }
        suffix = suffix_map.get(self.peft_method, self.peft_method.replace('-', '_'))
        return self.output_dir / "checkpoints" / f"user_{user_id}_{suffix}.pt"

    def _find_existing_checkpoint_path(self, user_id: str) -> Optional[Path]:
        """Find an existing checkpoint path, considering legacy filenames"""
        candidate = self._get_user_checkpoint_path(user_id)
        if candidate.exists():
            return candidate

        if self.peft_method == "lora":
            legacy_path = self.output_dir / "checkpoints" / f"user_{user_id}_lora.pt"
            if legacy_path.exists():
                return legacy_path

        return None

    def _checkpoint_exists(self, user_id: str) -> bool:
        """Check if an adapter checkpoint exists for the given user"""
        checkpoint_path = self._find_existing_checkpoint_path(user_id)
        exists = checkpoint_path is not None
        logger.debug(
            f"Checkpoint check for user {user_id}: {checkpoint_path} -> {'exists' if exists else 'not found'}"
        )
        return exists
    
    def _save_adapter_checkpoint(self, user_id: str, adapter_data: Dict) -> None:
        """Save adapter checkpoint to disk"""
        checkpoint_path = self._get_user_checkpoint_path(user_id)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        adapter_method = adapter_data.get('peft_method', self.peft_method)
        adapter_config = adapter_data.get('peft_config') or adapter_data.get('lora_config')

        if adapter_config is None:
            raise ValueError("Adapter config missing when attempting to save checkpoint")

        if isinstance(adapter_config, dict):
            adapter_config = self._config_from_dict(adapter_config, adapter_method)

        checkpoint = {
            'adapter_weights': adapter_data['adapter_weights'],
            'peft_config': adapter_config,
            'peft_method': adapter_method,
            'user_id': user_id,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Backward compatibility for legacy loaders expecting LoRA dicts
        if adapter_method == 'lora':
            fields = getattr(adapter_config, '__dataclass_fields__', {}).keys()
            checkpoint['lora_config_dict'] = {
                field: getattr(adapter_config, field)
                for field in fields
            }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved {adapter_method} checkpoint for user {user_id} to: {checkpoint_path}")

    def _load_adapter_checkpoint(self, user_id: str) -> Optional[Dict]:
        """Load adapter checkpoint from disk"""
        checkpoint_path = self._find_existing_checkpoint_path(user_id)

        if checkpoint_path is None:
            logger.debug(f"No checkpoint found for user {user_id} (method={self.peft_method})")
            return None

        try:
            logger.info(f"Attempting to load checkpoint for user {user_id} from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Validate checkpoint structure
            required_keys = ['adapter_weights', 'user_id']
            for key in required_keys:
                if key not in checkpoint:
                    logger.error(f"Invalid checkpoint for user {user_id}: missing key '{key}'")
                    return None

            # Log checkpoint info
            logger.info(f"Checkpoint contains {len(checkpoint['adapter_weights'])} adapter parameters")
            logger.info(f"Checkpoint timestamp: {checkpoint.get('timestamp', 'Unknown')}")

            adapter_method = checkpoint.get('peft_method', 'lora')
            adapter_config = checkpoint.get('peft_config')

            if adapter_config is None and 'lora_config_dict' in checkpoint:
                adapter_config = self._config_from_dict(checkpoint['lora_config_dict'], 'lora')
                adapter_method = 'lora'

            if isinstance(adapter_config, dict):
                adapter_config = self._config_from_dict(adapter_config, adapter_method)

            if adapter_config is None:
                logger.error(f"Adapter config missing or invalid in checkpoint for user {user_id}")
                return None

            adapter_data = {
                'adapter_weights': checkpoint['adapter_weights'],
                'peft_config': adapter_config,
                'peft_method': adapter_method,
                'user_id': user_id
            }

            logger.info(f"Successfully loaded {adapter_method} checkpoint for user {user_id}")
            return adapter_data

        except Exception as e:
            logger.error(f"Error loading checkpoint for user {user_id}: {e}")
            logger.info(f"Will train from scratch for user {user_id}")
            return None

    def train_and_evaluate(self):
        """Main training and evaluation loop"""
        logger.info("Starting OPPU training and evaluation")
        
        # Count existing checkpoints if enabled
        if self.use_checkpoints:
            existing_checkpoints = 0
            for user_data in self.test_data:
                if self._checkpoint_exists(user_data['user_id']):
                    existing_checkpoints += 1
            logger.info(f"Found {existing_checkpoints}/{len(self.test_data)} existing checkpoints")
        
        all_results = []
        user_performance_summaries = []
        
        for user_data in tqdm(self.test_data, desc="Processing users"):
            user_id = user_data['user_id']
            history = user_data.get('history', [])
            training_time_seconds = 0.0
            inference_time_seconds = 0.0
            
            # Check if a checkpoint exists for this user (if checkpoints are enabled)
            adapter_data = None
            if self.use_checkpoints and self._checkpoint_exists(user_id):
                logger.info(f"Found existing checkpoint for user {user_id}, attempting to load")
                adapter_data = self._load_adapter_checkpoint(user_id)
                if adapter_data:
                    logger.info(
                        f"Successfully loaded checkpoint for user {user_id}, skipping adapter training"
                    )
                else:
                    logger.warning(
                        f"Failed to load checkpoint for user {user_id}, will train from scratch"
                    )
            else:
                if self.use_checkpoints:
                    logger.debug(f"No checkpoint found for user {user_id}, will train from scratch")
                else:
                    logger.debug(f"Checkpoints disabled, training user {user_id} from scratch")

            # If no checkpoint loaded, train from scratch
            if adapter_data is None:
                # Prepare training data from history
                training_data = self._prepare_training_data(history)

                # Train user-specific adapter in memory
                if training_data:
                    logger.info(f"Training {self.peft_method} adapter for user {user_id} from scratch")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    train_start_time = time.perf_counter()
                    adapter_data = self._train_user_adapter(user_id, training_data)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    training_time_seconds = time.perf_counter() - train_start_time

                    # Save checkpoint after training (if checkpoints are enabled)
                    if self.use_checkpoints and adapter_data:
                        self._save_adapter_checkpoint(user_id, adapter_data)
                else:
                    logger.info(f"No training data for user {user_id}, skipping training")

            # Evaluate on user queries with in-memory adapter weights
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_start_time = time.perf_counter()
            user_results = self._evaluate_user(user_data, adapter_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time_seconds = time.perf_counter() - inference_start_time
            all_results.extend(user_results)
            
            # Evaluate and save individual user performance
            user_performance_summary = self._evaluate_and_save_user_performance(
                user_id,
                user_results,
                training_time_seconds,
                inference_time_seconds,
            )
            user_performance_summaries.append(user_performance_summary)

            # Clean up adapter data from memory after evaluation
            if adapter_data:
                del adapter_data
                torch.cuda.empty_cache()
            
        # Save results
        results_file = self.output_dir / f"oppu_results_{self.full_dataset_name}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Total predictions: {len(all_results)}")
        
        # Compute final evaluation metrics
        final_metrics = self._compute_final_metrics(all_results)
        
        # Save evaluation metrics
        metrics_file = self.output_dir / f"oppu_metrics_{self.full_dataset_name}.json"
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
            
        # Save user performance summaries
        user_summaries_file = self.output_dir / f"user_performance_summaries_{self.full_dataset_name}.json"

        # Analyze user performance distribution
        performance_distribution = self._analyze_user_performance_distribution(user_performance_summaries)

        total_training_time = sum(
            float(summary.get("training_time_seconds", 0.0))
            for summary in user_performance_summaries
            if isinstance(summary, dict) and isinstance(summary.get("training_time_seconds"), (int, float))
        )
        total_inference_time = sum(
            float(summary.get("inference_time_seconds", 0.0))
            for summary in user_performance_summaries
            if isinstance(summary, dict) and isinstance(summary.get("inference_time_seconds"), (int, float))
        )

        performance_overview = {
            "dataset": self.full_dataset_name,
            "total_users": len(user_performance_summaries),
            "total_queries": len(all_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_metrics": final_metrics,
            "performance_distribution": performance_distribution,
            "total_training_time_seconds": total_training_time,
            "total_inference_time_seconds": total_inference_time,
            "user_summaries": user_performance_summaries
        }
        with open(user_summaries_file, 'w') as f:
            json.dump(performance_overview, f, indent=2)
            
        logger.info(f"Evaluation metrics saved to: {metrics_file}")
        logger.info(f"User performance summaries saved to: {user_summaries_file}")
        logger.info(f"Performance distribution: {performance_distribution}")
        logger.info(f"Final metrics: {final_metrics}")
        
        return {
            "raw_results": all_results,
            "metrics": final_metrics,
            "user_performance_summaries": user_performance_summaries,
            "results_file": str(results_file),
            "metrics_file": str(metrics_file),
            "user_summaries_file": str(user_summaries_file)
        }


def main():
    parser = argparse.ArgumentParser(description="OPPU Training for Text-to-PEFT adapters")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")  # Match baseline
    parser.add_argument("--data_path", type=str, default="./data_p13n")
    parser.add_argument("--dataset_name", type=str, nargs='+', required=True, 
                       help="Dataset name(s) with test type (e.g., LaMP_processed_movie_random_test LongLaMP_product_review_ood_test). Can specify multiple datasets.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_retrieved_examples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)  # Match baseline
    parser.add_argument("--learning_rate", type=float, default=8e-5)  # Match baseline
    parser.add_argument("--num_epochs", type=int, default=100)  # Match baseline
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)  # Match baseline
    parser.add_argument("--peft_method", type=str, default="lora",
                       help="Which PEFT method to use: lora (default), ia3, prompt_tuning")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of gradient accumulation steps to reduce memory usage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing during training (default: enabled)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", default=False,
                       help="Disable gradient checkpointing (overrides --gradient_checkpointing)")
    parser.add_argument("--access_token", type=str, default=None)
    parser.add_argument("--use_checkpoints", action="store_true", default=True,
                       help="Whether to save and load adapter checkpoints")
    parser.add_argument("--no_checkpoints", action="store_true", default=False,
                       help="Disable checkpoint loading/saving and force retraining")
    parser.add_argument("--data_version", type=str, default="auto",
                       help="Which data version to use: 'v2' for v2 format, 'v3' for v3 format, 'v4', 'v5', etc. for higher versions, or 'auto' to automatically use highest available version (default: auto)")
    
    args = parser.parse_args()
    
    # Validate data_version argument
    if args.data_version != 'auto' and not (args.data_version.startswith('v') and args.data_version[1:].isdigit()):
        logger.error(f"Invalid data_version '{args.data_version}'. Must be 'auto' or version format like 'v2', 'v3', 'v4', etc.")
        return
    
    # Handle checkpoint arguments
    use_checkpoints = args.use_checkpoints and not args.no_checkpoints
    # Handle gradient checkpointing arguments
    enable_gradient_checkpointing = args.gradient_checkpointing and not args.no_gradient_checkpointing
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Log checkpoint settings
    if use_checkpoints:
        logger.info("Checkpoint mode enabled: Will save/load adapter checkpoints")
    else:
        logger.info("Checkpoint mode disabled: Will retrain all adapters")
    
    logger.info(f"Data version setting: {args.data_version}")
    
    # Iterate through each dataset
    for full_dataset_name in args.dataset_name:
        logger.info(f"Starting training and evaluation for dataset: {full_dataset_name}")
        
        # Create dataset-specific output directory using full name
        dataset_output_dir = os.path.join(args.output_dir, full_dataset_name)
        
        try:
            # Create trainer and run for this dataset
            trainer = OPPUTrainer(
                model_name=args.model_name,
                data_path=args.data_path,
                dataset_name=full_dataset_name,
                output_dir=dataset_output_dir,
                max_length=args.max_length,
                num_retrieved_examples=args.num_retrieved_examples,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                peft_method=args.peft_method,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=enable_gradient_checkpointing,
                access_token=args.access_token,
                use_checkpoints=use_checkpoints,
                data_version=args.data_version,
            )
            
            trainer.train_and_evaluate()
            logger.info(f"Completed training and evaluation for dataset: {full_dataset_name}")
            
        except Exception as e:
            logger.error(f"Error processing dataset {full_dataset_name}: {str(e)}")
            continue
    
    logger.info(f"Completed processing all {len(args.dataset_name)} datasets")


if __name__ == "__main__":
    main() 
