import json
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse
from datasets import Dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def _to_plain_text(value: Any) -> str:
    """Ensure profile_text is a plain string.
    - Lists/tuples: join items with a space after casting to str
    - None: empty string
    - Other types: str(value)
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        try:
            return " ".join(str(v) for v in value if v is not None).strip()
        except Exception:
            return str(value)
    return str(value)

# -----------------------------------------------------------------------------
# Helpers for history label distribution stats
# -----------------------------------------------------------------------------
def _compute_history_label_distribution(task_name: str, history_items: List[Dict]) -> str:
    """
    Compute a flat string summarizing the distribution of labels in user history.

    Implemented for:
      - movie: uses 'tag'
      - news_cat: uses 'category'

    Returns a concise single-line string, e.g.:
      "total=10 | sci-fi=3 (30.0%), comedy=2 (20.0%), action=5 (50.0%)"

    Returns an empty string if no applicable history exists.
    """
    try:
        label_key = None
        if task_name == 'movie':
            label_key = 'tag'
        elif task_name == 'news_cat':
            label_key = 'category'

        if not label_key or not history_items:
            return ""

        counts = {}
        total = 0
        for item in history_items:
            if not isinstance(item, dict):
                continue
            label = item.get(label_key)
            if label is None:
                continue
            label_str = str(label).strip()
            if not label_str:
                continue
            counts[label_str] = counts.get(label_str, 0) + 1
            total += 1

        if total == 0 or not counts:
            return ""

        # Sort by count desc, then label asc for stability
        sorted_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        parts = []
        for lbl, cnt in sorted_counts:
            pct = (cnt / float(total)) * 100.0
            parts.append(f"{lbl}={cnt} ({pct:.1f}%)")
        return f"total={total} | " + ", ".join(parts)
    except Exception:
        # Be robust: never break dataset creation
        return ""

# Task-specific key definitions and standardized names
TASK_KEY_MAPPINGS = {
    # LaMP tasks
    'movie': {
        'key_order': ['description', 'tag'],
        'key_mapping': {
            'description': 'description',
            'tag': 'tag',
            # Additional fields that might exist
            # 'title': 'metadata',
            # 'year': 'metadata',
            # 'rating': 'metadata'
        },
        'display_names': {
            'description': 'Description',
            'tag': 'Tag'
            # 'metadata': 'Additional Info'
        }
    },
    'news_cat': {
        'key_order': ['article', 'category'],
        'key_mapping': {
            'text': 'article',
            'category': 'category'
         },
        'display_names': {
            'article': 'Article',
            'category': 'Category'
            # 'metadata': 'Article Info'
        }
    },
    'news_headline': {
        'key_order': ['article', 'headline'],
        'key_mapping': {
            'text': 'article',
            'title': 'headline'
            # 'date': 'metadata',
            # 'source': 'metadata',
            # 'author': 'metadata'
        },
        'display_names': {
            'article': 'Article',
            'headline': 'Headline'
            # 'metadata': 'Article Info'
        }
    },
    'citation': {
        'key_order': ['paper_title', 'citation'],
        'key_mapping': {
            'title': 'paper_title',
            'citation': 'citation'
            # 'authors': 'paper_metadata',
            # 'year': 'paper_metadata',
            # 'venue': 'paper_metadata',
            # 'abstract': 'paper_metadata'
        },
        'display_names': {
            'paper_title': 'Paper Title',
            'citation': 'Citation'
            # 'paper_metadata': 'Paper Details'
        }
    },
    'product': {
        'key_order': ['review', 'score'],
        'key_mapping': {
            'text': 'review',
            'score': 'score',
            # 'overall': 'score',
            # 'summary': 'summary',
            # 'product_id': 'product_id',
            # 'helpful': 'helpful'
        },
        'display_names': {
            'review': 'Review',
            'score': 'Score'
            # 'product_info': 'Product Details'
        }
    },
    'scholarly_title': {
        'key_order': ['abstract', 'title'],
        'key_mapping': {
            'abstract': 'abstract',
            'title': 'title',
            # 'authors': 'authors',
            # 'year': 'year',
            # 'venue': 'venue',
            # 'keywords': 'paper_metadata'
        },
        'display_names': {
            'abstract': 'Abstract',
            'title': 'Title'
            # 'paper_metadata': 'Paper Details'
        }
    },
    'tweet': {
        'key_order': ['tweet'],
        'key_mapping': {
            'text': 'tweet',
            # 'likes': 'engagement_metrics',
            # 'retweets': 'engagement_metrics',
            # 'replies': 'engagement_metrics',
            # 'user_id': 'user_info',
            # 'timestamp': 'user_info',
            # 'hashtags': 'user_info'
        },
        'display_names': {
            'tweet': 'Tweet'
            # 'engagement_metrics': 'Engagement',
            # 'user_info': 'User Details'
        }
    },
    # LongLaMP tasks
    'product_review': {
        'key_order': ['product_description', 'rating', "review_summary"],
        'key_mapping': {
            'description': 'product_description',
            'rating': 'rating',
            'summary': 'review_summary',
            # 'helpful': 'helpful',
            # 'product_id': 'product_id',
            # 'reviewer_id': 'reviewer_id'
        },
        'display_names': {
            'product_description': 'Product Description',
            'rating': 'Rating',
            'review_summary': 'Review Summary'
            # 'review_metadata': 'Review Details'
        }
    },
    'abstract_generation': {
        'key_order': ['paper_title', 'abstract'],
        'key_mapping': {
            'title': 'paper_title',
            'abstract': 'abstract',
            # 'authors': 'paper_metadata',
            # 'year': 'paper_metadata',
            # 'venue': 'paper_metadata',
            # 'keywords': 'paper_metadata'
        },
        'display_names': {
            'paper_title': 'Paper Title',
            'abstract': 'Abstract'
            # 'paper_metadata': 'Paper Details'
        }
    },
    'topic_writing': {
        'key_order': ['topic', 'post'],
        'key_mapping': {
            'summary': 'topic',
            'content': 'post',
            # 'author': 'author',
            # 'date': 'date',
            # 'genre': 'genre',
            # 'length': 'writing_metadata'
        },
        'display_names': {
            'topic': 'Topic',
            'post': 'Post'
            # 'writing_metadata': 'Writing Details'
        }
    },
    # Default/fallback mapping for unknown tasks
    'default': {
        'key_order': ['primary_content', 'secondary_content', 'metadata'],
        'key_mapping': {},  # Will use original keys
        'display_names': {
            'primary_content': 'Primary Content',
            'secondary_content': 'Secondary Content', 
            'metadata': 'Additional Details'
        }
    }
}

def get_task_key_mapping(task_name: str) -> Dict:
    """Get the key mapping configuration for a specific task"""
    return TASK_KEY_MAPPINGS.get(task_name, TASK_KEY_MAPPINGS['default'])

def parse_version_number(version_str: str) -> int:
    """Parse version string (e.g., 'v3', 'v4') to get version number"""
    if version_str.startswith('v') and version_str[1:].isdigit():
        return int(version_str[1:])
    return 0

def find_available_versions(data_dir: str) -> List[str]:
    """Find all available version directories in data_dir"""
    available_versions = []
    for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)) and item.startswith('v') and item[1:].isdigit():
            available_versions.append(item)
    # Sort by version number
    available_versions.sort(key=parse_version_number, reverse=True)
    return available_versions

def determine_version_to_use(data_dir: str, data_version: str) -> tuple:
    """
    Determine which version directory to use and format type.
    Returns (version_dir_path, format_type) where format_type is 'v2' or 'v3+'
    """
    if data_version == 'auto':
        # Find highest available version
        available_versions = find_available_versions(data_dir)
        if not available_versions:
            return None, None
        
        # Use the highest version available
        chosen_version = available_versions[0]
        version_dir = os.path.join(data_dir, chosen_version)
        
        # Determine format type
        version_num = parse_version_number(chosen_version)
        format_type = 'v2' if version_num == 2 else 'v3+'
        
        print(f"Auto-detected version: {chosen_version} (using {format_type} format)")
        return version_dir, format_type
    else:
        # Use specified version
        version_dir = os.path.join(data_dir, data_version)
        if not os.path.exists(version_dir):
            return None, None
        
        # Determine format type
        version_num = parse_version_number(data_version)
        if version_num == 2:
            format_type = 'v2'
        elif version_num >= 3:
            format_type = 'v3+'
        else:
            # Invalid version format
            return None, None
        
        print(f"Using specified version: {data_version} (using {format_type} format)")
        return version_dir, format_type

def standardize_history_item_keys(item: Dict, task_name: str) -> Dict:
    """Standardize the keys of a history item based on task-specific mapping, only including keys from key_order"""
    if not isinstance(item, dict):
        return item
    
    task_config = get_task_key_mapping(task_name)
    key_mapping = task_config['key_mapping']
    key_order = task_config['key_order']
    
    # If no specific mapping, return original item with only keys that exist
    if not key_mapping:
        return {k: v for k, v in item.items() if k in key_order}
    
    standardized_item = {}
    
    # Only process keys that are mapped AND result in keys that are in key_order
    for original_key, value in item.items():
        if original_key in key_mapping:
            standard_key = key_mapping[original_key]
            # Only include if the standardized key is in the key_order
            if standard_key in key_order:
                standardized_item[standard_key] = value
    
    return standardized_item

def format_standardized_history_item(item: Dict, task_name: str, truncate_values: bool = True, max_value_length: int = 800) -> str:
    """Format a standardized history item with proper ordering and display names, only including keys from key_order"""
    if not isinstance(item, dict):
        return str(item)[:200] if truncate_values else str(item)
    
    task_config = get_task_key_mapping(task_name)
    key_order = task_config['key_order']
    display_names = task_config['display_names']
    
    entry_parts = []
    
    # Process ONLY keys in the defined order (no other keys)
    for standard_key in key_order:
        if standard_key in item and item[standard_key] is not None:
            value = item[standard_key]
            value_str = str(value).strip()
            if value_str:
                if truncate_values and len(value_str) > max_value_length:
                    value_str = value_str[:max_value_length] + "..."
                display_name = display_names.get(standard_key, standard_key.replace('_', ' ').title())
                entry_parts.append(f"{display_name}: {value_str}")
    
    if entry_parts:
        return " | ".join(entry_parts)
    else:
        return f"User activity: {str(item)[:800]}" if truncate_values else str(item)

def estimate_token_count(text):
    """
    Rough estimation of token count for a given text.
    Uses the approximation that 1 token ≈ 4 characters for English text.
    """
    return len(text) // 4

def detect_history_format(sample_history_items, limit=10, task_name='default'):
    """Detect the format of history items by examining a sample - now uses task-specific standardization"""
    if not sample_history_items:
        return []
    
    # Return the standardized key order for the task
    task_config = get_task_key_mapping(task_name)
    return task_config['key_order']

def format_history_item(item, detected_keys=None, task_name='default'):
    """Format a single history item using task-specific standardization with value truncation"""
    if not isinstance(item, dict):
        return ""
    
    # Standardize the item keys first
    standardized_item = standardize_history_item_keys(item, task_name)
    
    # Format using the standardized approach
    return format_standardized_history_item(standardized_item, task_name, truncate_values=True, max_value_length=800)

def format_history_item_for_retrieval(item, detected_keys=None, task_name='default'):
    """Format a single history item for BM25 retrieval without truncation using task-specific standardization"""
    if not isinstance(item, dict):
        return str(item)
    
    # Standardize the item keys first
    standardized_item = standardize_history_item_keys(item, task_name)
    
    # Format using the standardized approach without truncation
    return format_standardized_history_item(standardized_item, task_name, truncate_values=True, max_value_length=800)

def bm25_profile_retrieval(history_data: List[Dict], query_text: str, k: int = 1, task_name: str = 'default', max_context_length: int = 28000) -> str:
    """Use BM25 to retrieve the most relevant history item for the given query"""
    if not history_data or not query_text:
        return ""
    
    # Get the standardized format for the task
    detected_keys = detect_history_format(history_data, task_name=task_name)
    
    # Format all history items without truncation for retrieval
    history_list = []
    for item in history_data:
        formatted_item = format_history_item_for_retrieval(item, detected_keys, task_name=task_name)
        if formatted_item:
            history_list.append(formatted_item)
    
    if not history_list:
        return ""
    
    # Tokenize corpus by splitting on spaces
    tokenized_corpus = [doc.split(" ") for doc in history_list]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tokenize query and retrieve top k results
    tokenized_query = query_text.split(' ')
    retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)
    
    # Return the retrieved items joined with separator (or empty string if no results)
    if retrieved_history:
        if k == 1:
            result = retrieved_history[0]
        else:
            # Join multiple retrieved items with double newline separator
            result = "\n\n".join(retrieved_history)
        
        # Truncate result to fit within context length
        estimated_tokens = estimate_token_count(result)
        if estimated_tokens > max_context_length:
            # Reserve some buffer for safety (10% of context length)
            target_chars = int((max_context_length * 0.9) * 4)  # Convert tokens to chars
            if len(result) > target_chars:
                result = result[:target_chars] + "..."
        
        return result
    else:
        return ""

def bm25_profile_retrieval_batch(history_data: List[Dict], query_text: str, k_values: List[int], task_name: str = 'default', max_context_length: int = 28000) -> Dict[int, str]:
    """
    Use BM25 to retrieve the most relevant history items for the given query for multiple k values.
    This is more efficient than calling bm25_profile_retrieval multiple times as it computes scores once
    and reuses them for different k values.
    
    Returns:
        Dictionary mapping k values to their corresponding retrieved history strings
    """
    # Initialize result dictionary with empty strings
    results = {k: "" for k in k_values}
    
    if not history_data or not query_text or not k_values:
        return results
    
    # Get the standardized format for the task
    detected_keys = detect_history_format(history_data, task_name=task_name)
    
    # Format all history items without truncation for retrieval
    history_list = []
    for item in history_data:
        formatted_item = format_history_item_for_retrieval(item, detected_keys, task_name=task_name)
        if formatted_item:
            history_list.append(formatted_item)
    
    if not history_list:
        return results
    
    # Tokenize corpus by splitting on spaces
    tokenized_corpus = [doc.split(" ") for doc in history_list]
    
    # Create BM25 index once
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tokenize query and get scores for all documents
    tokenized_query = query_text.split(' ')
    scores = bm25.get_scores(tokenized_query)
    
    # Get the maximum k value to determine how many top results we need
    max_k = max(k_values)
    
    # Get indices of top max_k documents sorted by score (descending)
    top_indices = np.argsort(scores)[::-1][:max_k]
    
    # For each k value, get the top k results
    for k in k_values:
        if len(top_indices) > 0:
            # Use all available documents if k exceeds available documents
            if k <= len(top_indices):
                top_k_indices = top_indices[:k]
            else:
                top_k_indices = top_indices
            retrieved_history = [history_list[i] for i in top_k_indices]
            
            if retrieved_history:
                if len(retrieved_history) == 1:
                    result = retrieved_history[0]
                else:
                    # Join multiple retrieved items with double newline separator
                    result = "\n\n".join(retrieved_history)
                
                # Truncate result to fit within context length
                estimated_tokens = estimate_token_count(result)
                if estimated_tokens > max_context_length:
                    # Reserve some buffer for safety (10% of context length)
                    target_chars = int((max_context_length * 0.9) * 4)  # Convert tokens to chars
                    if len(result) > target_chars:
                        result = result[:target_chars] + "..."
                
                results[k] = result
    
    return results

def truncate_history_to_fit_context(history_items, max_context_length, task_name='default'):
    """
    Truncate history values within items to fit within context length.
    First try truncating individual values, then remove items from beginning if still too long.
    Returns the processed history that fits within the context limit.
    """
    if not history_items or max_context_length <= 0:
        return history_items
    
    # First, quick check if we even need truncation
    total_chars = sum(len(str(item)) for item in history_items)
    estimated_tokens = total_chars // 4
    
    # If it's already small enough, return as is
    if estimated_tokens <= max_context_length * 0.5:  # Conservative estimate
        return history_items
    
    # Get the standardized format for the task
    detected_keys = detect_history_format(history_items, task_name=task_name)
    
    # First, try truncating values within items with progressively smaller limits
    for max_value_length in [700, 500, 300]:
        # Create truncated versions of all history items
        truncated_items = []
        total_length = 0
        
        for item in history_items:
            if isinstance(item, dict):
                truncated_item = {}
                item_length = 0
                
                for key, value in item.items():
                    if value is not None:
                        value_str = str(value).strip()
                        if len(value_str) > max_value_length:
                            truncated_value = value_str[:max_value_length] + "..."
                        else:
                            truncated_value = value_str
                        truncated_item[key] = truncated_value
                        item_length += len(truncated_value) + len(key) + 10  # Rough estimate
                    else:
                        truncated_item[key] = value
                
                truncated_items.append(truncated_item)
                total_length += item_length
            else:
                truncated_items.append(item)
                total_length += len(str(item))
        
        # Quick token estimate without building full string
        estimated_tokens = total_length // 4 + 1000  # Add buffer for formatting
        
        if estimated_tokens <= max_context_length * 0.9:
            return truncated_items
    
    # If truncating values isn't enough, progressively remove items from beginning
    min_value_length = 300
    for start_idx in range(len(history_items)):
        remaining_items = history_items[start_idx:]
        
        # Quick estimate
        total_chars = sum(
            sum(min(len(str(v)), min_value_length) for v in item.values() if v is not None)
            if isinstance(item, dict) else len(str(item))
            for item in remaining_items
        )
        estimated_tokens = total_chars // 4 + 100
        
        if estimated_tokens <= max_context_length * 0.9:
            # Apply minimal truncation to remaining items
            truncated_items = []
            for item in remaining_items:
                if isinstance(item, dict):
                    truncated_item = {}
                    for key, value in item.items():
                        if value is not None:
                            value_str = str(value).strip()
                            if len(value_str) > min_value_length:
                                truncated_item[key] = value_str[:min_value_length] + "..."
                            else:
                                truncated_item[key] = value
                        else:
                            truncated_item[key] = value
                    truncated_items.append(truncated_item)
                else:
                    truncated_items.append(item)
            return truncated_items
    
    # If even a single history item is too large, return empty list
    print(f"Warning: Even single history item exceeds context limit. Using empty history.")
    return []

def load_profiles(profile_file_path: str) -> Dict[str, str]:
    """Load profiles from JSON file"""
    if not os.path.exists(profile_file_path):
        print(f"Profile file not found: {profile_file_path}")
        return {}
    
    with open(profile_file_path, 'r') as f:
        profiles = json.load(f)
    
    # Flatten the profile structure to get user_id -> profile mapping
    user_profiles = {}
    for split_type, users in profiles.items():
        for user_id, profile_text in users.items():
            if user_id not in user_profiles:
                user_profiles[user_id] = profile_text
    
    return user_profiles

def deduplicate_train_data(train_data: List[Dict], test_data_list: List[List[Dict]]) -> List[Dict]:
    """Remove samples from train_data if their user_id appears in any of the test datasets"""
    if not train_data:
        return train_data
    
    # Collect all user_ids from test datasets
    test_user_ids = set()
    for test_data in test_data_list:
        for item in test_data:
            if 'user_id' in item:
                test_user_ids.add(item['user_id'])
    
    # Filter out train samples with user_ids that appear in test sets
    deduplicated_train = [item for item in train_data if item.get('user_id') not in test_user_ids]
    
    removed_count = len(train_data) - len(deduplicated_train)
    if removed_count > 0:
        print(f"Removed {removed_count} overlapping samples from train data (users also in test sets)")
    
    return deduplicated_train

# def load_all_generated_profiles(generated_profile_dir: str) -> Dict[str, str]:
#     """Load all generated profiles from JSON files, excluding ood/test/random files."""
#     print("Loading generated profiles...")
    
#     if not os.path.exists(generated_profile_dir):
#         print(f"Generated profile directory not found: {generated_profile_dir}")
#         return {}
    
#     # Get all profile files
#     profile_files = []
#     for file in os.listdir(generated_profile_dir):
#         if file.endswith('.json'):
#             profile_files.append(os.path.join(generated_profile_dir, file))
    
#     # Filter out files with 'ood', 'test', or 'random' in their names
#     filtered_files = []
#     for profile_file in profile_files:
#         filename = os.path.basename(profile_file).lower()
#         if not any(keyword in filename for keyword in ['ood', 'test', 'random']):
#             filtered_files.append(profile_file)
    
#     print(f"Found {len(profile_files)} profile files, using {len(filtered_files)} after filtering")
    
#     all_profiles = {}
#     for profile_file in filtered_files:
#         print(f"Loading profiles from {os.path.basename(profile_file)}")
        
#         try:
#             with open(profile_file, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
            
#             # Handle different JSON structures
#             if 'data' in data:
#                 # Structure: {"data": {"user_id": "profile_text", ...}}
#                 profiles = data['data']
#             else:
#                 # Structure: {"dataset_name": {"user_id": "profile_text", ...}, ...}
#                 profiles = {}
#                 for dataset_name, user_profiles in data.items():
#                     if isinstance(user_profiles, dict):
#                         profiles.update(user_profiles)
            
#             # Add profiles to the main dictionary
#             for user_id, profile_text in profiles.items():
#                 if user_id in all_profiles:
#                     print(f"Warning: Duplicate user_id found: {user_id}")
#                 all_profiles[user_id] = profile_text
                        
#         except Exception as e:
#             print(f"Error loading {profile_file}: {str(e)}")
#             continue
            
#     print(f"Loaded {len(all_profiles)} total user profiles")
#     return all_profiles

def extract_recent_user_profile(history_data: List[Dict], max_context_length: int = 28000, task_name: str = 'default') -> str:
    """Extract most recent user profile from interaction history that fits in context length"""
    if not history_data:
        return ""
    
    # Use the new truncation logic to fit within context length
    truncated_history = truncate_history_to_fit_context(history_data, max_context_length, task_name=task_name)
    
    if not truncated_history:
        return ""
    
    # Get the standardized format for the task
    detected_keys = detect_history_format(truncated_history, task_name=task_name)
    
    # Build profile from truncated history
    profile_parts = []
    for item in truncated_history:
        if isinstance(item, dict):
            item_text = format_history_item(item, detected_keys, task_name=task_name)
            if item_text:
                profile_parts.append(item_text)
    
    if not profile_parts:
        return ""
    
    # Build the recent profile - use different separator between history items
    recent_profile = "Recent User Activity:\n" + " || ".join(profile_parts)
    
    # Final validation: ensure the formatted profile fits within context
    profile_tokens = estimate_token_count(recent_profile)
    if profile_tokens > max_context_length:
        # If still too long, truncate the profile string directly
        # Leave some buffer for safety
        target_chars = (max_context_length - 100) * 4  # Reserve 100 tokens buffer
        if len(recent_profile) > target_chars:
            recent_profile = recent_profile[:target_chars] + "..."
    
    return recent_profile

def extract_recent_conversation_profile(conversations: List[Dict], max_context_length: int = 32768, task_name: str = 'default') -> str:
    """Extract recent user profile from PRISM conversation data"""
    if not conversations:
        return ""
    
    # Use the new truncation logic to fit within context length
    truncated_conversations = truncate_history_to_fit_context(conversations, max_context_length, task_name=task_name)
    
    if not truncated_conversations:
        return ""
    
    # Get the standardized format for the task
    detected_keys = detect_history_format(truncated_conversations, task_name=task_name)
    
    # Build profile from truncated conversations
    profile_parts = []
    for conv in truncated_conversations:
        if isinstance(conv, dict):
            conv_text = format_history_item(conv, detected_keys, task_name=task_name)
            if conv_text:
                profile_parts.append(conv_text)
    
    if not profile_parts:
        return ""
    
    recent_profile = "Recent Conversations:\n" + " || ".join(profile_parts)
    
    # Final validation: ensure the formatted profile fits within context
    profile_tokens = estimate_token_count(recent_profile)
    if profile_tokens > max_context_length:
        # If still too long, truncate the profile string directly
        # Leave some buffer for safety
        target_chars = (max_context_length - 100) * 4  # Reserve 100 tokens buffer
        if len(recent_profile) > target_chars:
            recent_profile = recent_profile[:target_chars] + "..."
    
    return recent_profile

def save_hf_dataset(data: List[Dict], output_dir: str):
    """Save dataset in Hugging Face format"""
    if not data:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset from list of dictionaries
    dataset = Dataset.from_list(data)
    
    # Save in Hugging Face format
    dataset.save_to_disk(output_dir)

def format_lamp_history_as_training_sample(task_name: str, history_item: Dict, user_id: str, history_idx: int, previous_history: List[Dict] = None, profile_context_length: int = 28000, generated_profiles: Dict[str, str] = None, include_generated_profile: bool = False, k: List[int] = [1, 2, 4]) -> Dict:
    """Format a LaMP history item as a training sample following the query template structure"""
    
    if not previous_history:
        previous_history = []
    
    if not generated_profiles:
        generated_profiles = {}
    
    # Extract recent profile from previous history items
    recent_profile = ""
    if previous_history:
        recent_profile = extract_recent_user_profile(previous_history, max_context_length=profile_context_length, task_name=task_name)
    
    # Set profile_text to only contain generated profile
    profile_text = ""
    if include_generated_profile and user_id in generated_profiles:
        profile_text = generated_profiles[user_id]
    
    # Format history item based on task type
    if task_name == 'movie':
        # Format like movie query: "Which tag does this movie relate to... description: {description}"
        description = history_item.get('description', '')
        tag = history_item.get('tag', '')
        if description and tag:
            input_text = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {description}"
            output_text = tag
        else:
            return None
            
    elif task_name == 'news_cat':
        # Format like news category query
        text = history_item.get('text', '') or history_item.get('article', '')
        category = history_item.get('category', '')
        if text and category:
            input_text = f"Which category does this article relate to among the following categories? Just answer with the category name without further explanation. categories: [travel, education, parents, style & beauty, entertainment, food & drink, science & technology, business, sports, healthy living, women, politics, crime, culture & arts, religion] article: {text}"
            output_text = category
        else:
            return None
            
    elif task_name == 'news_headline':
        # Format like news headline query
        text = history_item.get('text', '') or history_item.get('article', '')
        title = history_item.get('title', '')
        if text and title:
            input_text = f"Generate a headline for the following article: {text}"
            output_text = title
        else:
            return None
            
    elif task_name == 'citation':
        # Format like citation query (not used for history enrichment)
        title = history_item.get('title', '')
        citation = history_item.get('citation', '')
        if title and citation:
            input_text = f"Given the paper title, generate a citation for this paper. paper title: {title}"
            output_text = citation
        else:
            return None
            
    elif task_name == 'product':
        # Format like product rating query
        text = history_item.get('text', '') or history_item.get('reviewText', '')
        score = history_item.get('score', '') or history_item.get('overall', '')
        if text and score:
            input_text = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {text}"
            output_text = str(score)
        else:
            return None
            
    elif task_name == 'scholarly_title':
        # Format like scholarly title query
        abstract = history_item.get('abstract', '')
        title = history_item.get('title', '')
        if abstract and title:
            input_text = f"Generate a title for the following abstract of a paper: {abstract}"
            output_text = title
        else:
            return None
            
    elif task_name == 'tweet':
        # Format like tweet query (not used for history enrichment)
        text = history_item.get('text', '')
        if text:
            input_text = f"Paraphrase the following tweet without any explanation before or after it: {text}"
            output_text = text
        else:
            return None
    else:
        return None
    
    if not input_text or not output_text:
        return None
        
    question_id = f"lamp_{task_name}_{user_id}_history_{history_idx}"
    
    dataset_item = {
        'user_id': user_id,
        'question_id': question_id,
        'profile_text': profile_text,
        'profile_all_history': recent_profile,
        'input': input_text,
        'output': output_text
    }

    # For classification tasks, include history label distribution statistics
    if task_name in ['movie', 'news_cat']:
        # Use only previous history (before the current history item)
        history_stat = _compute_history_label_distribution(task_name, previous_history)
        dataset_item['history_stat'] = history_stat
    
    # Add profile retrieval for each k value (using previous history) with batch retrieval for efficiency
    if previous_history:
        retrieval_results = bm25_profile_retrieval_batch(previous_history, input_text, k, task_name=task_name, max_context_length=profile_context_length)
        for k_val in k:
            dataset_item[f'profile_retrieval_k{k_val}'] = retrieval_results[k_val]
    else:
        for k_val in k:
            dataset_item[f'profile_retrieval_k{k_val}'] = ""

    return dataset_item

def format_longlamp_history_as_training_sample(task_name: str, history_item: Dict, user_id: str, history_idx: int, previous_history: List[Dict] = None, profile_context_length: int = 28000, generated_profiles: Dict[str, str] = None, include_generated_profile: bool = False, k: List[int] = [1, 2, 4]) -> Dict:
    """Format a LongLaMP history item as a training sample following the query template structure"""
    
    if not previous_history:
        previous_history = []
    
    if not generated_profiles:
        generated_profiles = {}
    
    # Extract recent profile from previous history items
    recent_profile = ""
    if previous_history:
        recent_profile = extract_recent_user_profile(previous_history, max_context_length=profile_context_length, task_name=task_name)
    
    # Set profile_text to only contain generated profile
    profile_text = ""
    if include_generated_profile and user_id in generated_profiles:
        profile_text = generated_profiles[user_id]
    
    # Format history item based on task type
    if task_name == 'product_review':
        # Format like product review generation query
        description = history_item.get('description', '')
        overall = history_item.get('overall', '')
        summary = history_item.get('summary', '')
        review_text = history_item.get('reviewText', '')
        
        if description and overall and summary and review_text:
            input_text = f'Generate the review text written by a reviewer who has a given an overall rating of "{overall}" for a product with description "{description}". The summary of the review text is "{summary}".'
            output_text = review_text
        else:
            return None
            
    elif task_name == 'abstract_generation':
        # Format like abstract generation query
        title = history_item.get('title', '')
        abstract = history_item.get('abstract', '')
        
        if title and abstract:
            input_text = f"Generate an abstract for a paper with the title: {title}"
            output_text = abstract
        else:
            return None
            
    elif task_name == 'topic_writing':
        # Format like topic writing query
        summary = history_item.get('summary', '')
        content = history_item.get('content', '')
        
        if summary and content:
            input_text = f"Write content based on the following topic: {summary}"
            output_text = content
        else:
            return None
    else:
        return None
    
    if not input_text or not output_text:
        return None
        
    question_id = f"longlamp_{task_name}_{user_id}_history_{history_idx}"
    
    dataset_item = {
        'user_id': user_id,
        'question_id': question_id,
        'profile_text': profile_text,
        'profile_all_history': recent_profile,
        'input': input_text,
        'output': output_text
    }
    
    # Add profile retrieval for each k value (using previous history) with batch retrieval for efficiency
    if previous_history:
        retrieval_results = bm25_profile_retrieval_batch(previous_history, input_text, k, task_name=task_name, max_context_length=profile_context_length)
        for k_val in k:
            dataset_item[f'profile_retrieval_k{k_val}'] = retrieval_results[k_val]
    else:
        for k_val in k:
            dataset_item[f'profile_retrieval_k{k_val}'] = ""
    
    return dataset_item



def create_ec_dataset(data_dir: str, output_dir: str, splits: str = 'both', data_version: str = 'auto'):
    """Convert EC data to HF format with combined train and separate test datasets"""
    print("Processing EC dataset...")
    
    # Determine which version to use
    version_dir, format_type = determine_version_to_use(data_dir, data_version)
    
    if version_dir is None or format_type is None:
        print(f"Could not find or determine version directory in: {data_dir}")
        return
    
    import glob
    
    # Collect all train and test data
    train_data = []
    random_test_data = []
    ood_test_data = []
    
    # Determine which splits to load based on the splits parameter
    load_random = splits in ['random', 'both']
    load_ood = splits in ['ood', 'both']
    
    # Process train splits
    train_splits = []
    if load_random:
        train_splits.append('random_train')
    if load_ood:
        train_splits.append('ood_train')
        
    for split in train_splits:
        pattern = os.path.join(version_dir, f"*_{split}.jsonl")
        matching_files = glob.glob(pattern)
        
        # If no specific train split found, try generic train pattern
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            # Try fallback to generic train pattern
            generic_pattern = os.path.join(version_dir, "*_train.jsonl")
            matching_files = glob.glob(generic_pattern)
            if matching_files:
                print(f"Found generic train file: {matching_files[0]}")
            else:
                print(f"No train files found at all")
                continue
        
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")
        
        with open(input_file, 'r') as f:
            for idx, line in enumerate(tqdm(f, desc=f"Processing EC {split}")):
                if line.strip():
                    item = json.loads(line)
                    
                    dataset_item = {
                        'user_id': item['user_id'],
                        'question_id': f"ec_{item['user_id']}_{idx}_{split}",
                        'profile_text': _to_plain_text(item.get('profile_text', "")),
                        'input': item['article'],
                        'output': item['essay']
                    }
                    train_data.append(dataset_item)
    
    # Process test splits
    test_splits = []
    if load_random:
        test_splits.append('random_test')
    if load_ood:
        test_splits.append('ood_test')
        
    for split in test_splits:
        pattern = os.path.join(version_dir, f"*_{split}.jsonl")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            continue
        
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")
        
        current_test_data = []
        with open(input_file, 'r') as f:
            for idx, line in enumerate(tqdm(f, desc=f"Processing EC {split}")):
                if line.strip():
                    item = json.loads(line)
                    
                    dataset_item = {
                        'user_id': item['user_id'],
                        'question_id': f"ec_{item['user_id']}_{idx}_{split}",
                        'profile_text': _to_plain_text(item.get('profile_text', "")),
                        'input': item['article'],
                        'output': item['essay']
                    }
                    # print(item.get('profile_text'))
                    # input()
                    current_test_data.append(dataset_item)
        
        # Store test data in appropriate list
        if split == 'random_test':
            random_test_data = current_test_data
        elif split == 'ood_test':
            ood_test_data = current_test_data
    
    # Deduplicate train data against test datasets
    test_datasets = [random_test_data, ood_test_data]
    deduplicated_train_data = deduplicate_train_data(train_data, test_datasets)
    
    # Save datasets
    if deduplicated_train_data:
        output_path = os.path.join(output_dir, "EC", "train")
        save_hf_dataset(deduplicated_train_data, output_path)
        print(f"Saved EC train dataset: {len(deduplicated_train_data)} samples")
    
    if random_test_data:
        output_path = os.path.join(output_dir, "EC", "random_test")
        save_hf_dataset(random_test_data, output_path)
        print(f"Saved EC random_test dataset: {len(random_test_data)} samples")
    
    if ood_test_data:
        output_path = os.path.join(output_dir, "EC", "ood_test")
        save_hf_dataset(ood_test_data, output_path)
        print(f"Saved EC ood_test dataset: {len(ood_test_data)} samples")

def create_opinionqa_dataset(data_dir: str, output_dir: str, splits: str = 'both', data_version: str = 'auto'):
    """Convert OpinionQA data to HF format with combined train and separate test datasets"""
    print("Processing OpinionQA dataset...")
    
    # Determine which version to use
    version_dir, format_type = determine_version_to_use(data_dir, data_version)
    
    if version_dir is None or format_type is None:
        print(f"Could not find or determine version directory in: {data_dir}")
        return
    
    import glob
    
    # Collect all train and test data
    train_data = []
    random_test_data = []
    ood_test_data = []
    
    # Determine which splits to load based on the splits parameter
    load_random = splits in ['random', 'both']
    load_ood = splits in ['ood', 'both']
    
    # Process train splits
    train_splits = []
    if load_random:
        train_splits.append('random_train')
    if load_ood:
        train_splits.append('ood_train')
        
    for split in train_splits:
        pattern = os.path.join(version_dir, f"*_{split}.jsonl")
        matching_files = glob.glob(pattern)
        
        # If no specific train split found, try generic train pattern
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            # Try fallback to generic train pattern
            generic_pattern = os.path.join(version_dir, "*_train.jsonl")
            matching_files = glob.glob(generic_pattern)
            if matching_files:
                print(f"Found generic train file: {matching_files[0]}")
            else:
                print(f"No train files found at all")
                continue
        
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")
        
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc=f"Processing OpinionQA {split}"):
                if line.strip():
                    item = json.loads(line)
                    
                    dataset_item = {
                        'user_id': item['user_id'],
                        'question_id': f"{item['question_id']}_{split}",
                        'profile_text': _to_plain_text(item.get('profile_text', "")),
                        'input': item['input'],
                        'output': item['output']
                    }
                    train_data.append(dataset_item)
    
    # Process test splits
    test_splits = []
    if load_random:
        test_splits.append('random_test')
    if load_ood:
        test_splits.append('ood_test')
        
    for split in test_splits:
        pattern = os.path.join(version_dir, f"*_{split}.jsonl")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            continue
        
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")
        
        current_test_data = []
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc=f"Processing OpinionQA {split}"):
                if line.strip():
                    item = json.loads(line)
                    
                    dataset_item = {
                        'user_id': item['user_id'],
                        'question_id': f"{item['question_id']}_{split}",
                        'profile_text': _to_plain_text(item.get('profile_text', "")),
                        'input': item['input'],
                        'output': item['output']
                    }
                    current_test_data.append(dataset_item)
        
        # Store test data in appropriate list
        if split == 'random_test':
            random_test_data = current_test_data
        elif split == 'ood_test':
            ood_test_data = current_test_data
    
    # Deduplicate train data against test datasets
    test_datasets = [random_test_data, ood_test_data]
    deduplicated_train_data = deduplicate_train_data(train_data, test_datasets)
    
    # Save datasets
    if deduplicated_train_data:
        output_path = os.path.join(output_dir, "OpinionQA", "train")
        save_hf_dataset(deduplicated_train_data, output_path)
        print(f"Saved OpinionQA train dataset: {len(deduplicated_train_data)} samples")
    
    if random_test_data:
        output_path = os.path.join(output_dir, "OpinionQA", "random_test")
        save_hf_dataset(random_test_data, output_path)
        print(f"Saved OpinionQA random_test dataset: {len(random_test_data)} samples")
    
    if ood_test_data:
        output_path = os.path.join(output_dir, "OpinionQA", "ood_test")
        save_hf_dataset(ood_test_data, output_path)
        print(f"Saved OpinionQA ood_test dataset: {len(ood_test_data)} samples")

def create_personalreddit_dataset(data_dir: str, output_dir: str, splits: str = 'both', data_version: str = 'auto'):
    """Convert PersonalReddit data to HF format with combined train and separate test datasets"""
    print("Processing PersonalReddit dataset...")
    
    # Determine which version to use
    version_dir, format_type = determine_version_to_use(data_dir, data_version)
    
    if version_dir is None or format_type is None:
        print(f"Could not find or determine version directory in: {data_dir}")
        return
    
    import glob
    
    # Collect all train and test data
    train_data = []
    random_test_data = []
    ood_test_data = []
    
    # Determine which splits to load based on the splits parameter
    load_random = splits in ['random', 'both']
    load_ood = splits in ['ood', 'both']
    
    # Process train splits
    train_splits = []
    if load_random:
        train_splits.append('random_train')
    if load_ood:
        train_splits.append('ood_train')
        
    for split in train_splits:
        pattern = os.path.join(version_dir, f"*_{split}.jsonl")
        matching_files = glob.glob(pattern)
        
        # If no specific train split found, try generic train pattern
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            # Try fallback to generic train pattern
            generic_pattern = os.path.join(version_dir, "*_train.jsonl")
            matching_files = glob.glob(generic_pattern)
            if matching_files:
                print(f"Found generic train file: {matching_files[0]}")
            else:
                print(f"No train files found at all")
                continue
        
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")
        
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc=f"Processing PersonalReddit {split}"):
                if line.strip():
                    item = json.loads(line)
                    
                    dataset_item = {
                        'user_id': item['user_id'],
                        'question_id': f"pr_{item['user_id']}_{split}",
                        'profile_text': _to_plain_text(item.get('profile_text', "")),
                        'input': item['input'],
                        'output': item['output']
                    }
                    train_data.append(dataset_item)
    
    # Process test splits
    test_splits = []
    if load_random:
        test_splits.append('random_test')
    if load_ood:
        test_splits.append('ood_test')
        
    for split in test_splits:
        pattern = os.path.join(version_dir, f"*_{split}.jsonl")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            continue
        
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")
        
        current_test_data = []
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc=f"Processing PersonalReddit {split}"):
                if line.strip():
                    item = json.loads(line)
                    
                    dataset_item = {
                        'user_id': item['user_id'],
                        'question_id': f"pr_{item['user_id']}_{split}",
                        'profile_text': _to_plain_text(item.get('profile_text', "")),
                        'input': item['input'],
                        'output': item['output']
                    }
                    current_test_data.append(dataset_item)
        
        # Store test data in appropriate list
        if split == 'random_test':
            random_test_data = current_test_data
        elif split == 'ood_test':
            ood_test_data = current_test_data
    
    # Deduplicate train data against test datasets
    test_datasets = [random_test_data, ood_test_data]
    deduplicated_train_data = deduplicate_train_data(train_data, test_datasets)
    
    # Save datasets
    if deduplicated_train_data:
        output_path = os.path.join(output_dir, "PersonalReddit", "train")
        save_hf_dataset(deduplicated_train_data, output_path)
        print(f"Saved PersonalReddit train dataset: {len(deduplicated_train_data)} samples")
    
    if random_test_data:
        output_path = os.path.join(output_dir, "PersonalReddit", "random_test")
        save_hf_dataset(random_test_data, output_path)
        print(f"Saved PersonalReddit random_test dataset: {len(random_test_data)} samples")
    
    if ood_test_data:
        output_path = os.path.join(output_dir, "PersonalReddit", "ood_test")
        save_hf_dataset(ood_test_data, output_path)
        print(f"Saved PersonalReddit ood_test dataset: {len(ood_test_data)} samples")

def create_lamp_dataset(data_dir: str, output_dir: str, generated_profile_dir: str, include_generated_profile: bool = True, k: List[int] = [1], profile_context_length: int = 28000, splits: str = 'both', data_version: str = 'auto'):
    """Convert LaMP data to HF format"""
    print("Processing LaMP dataset...")
    
    # Map of task names to their profile files
    task_profile_map = {
        'movie': 'LaMP_processed_movie_profiles.json',
        'news_cat': 'LaMP_processed_news_cat_profiles.json',
        'scholarly_title': 'LaMP_processed_scholarly_title_profiles.json',
        'tweet': 'LaMP_processed_tweet_profiles.json',
        'citation': 'LaMP_processed_citation_profiles.json',
        "news_headline": "LaMP_processed_news_headline_profiles.json",
        "product": "LaMP_processed_product_profiles.json"
    }
    
    # Determine which splits to include based on the splits parameter
    allowed_splits = []
    if splits == 'random':
        allowed_splits = ['random_train', 'random_test']
    elif splits == 'ood':
        allowed_splits = ['ood_train', 'ood_test']
    else:  # splits == 'both'
        allowed_splits = ['random_train', 'random_test', 'ood_train', 'ood_test']
        
    # Determine which version to use
    version_dir, format_type = determine_version_to_use(data_dir, data_version)
    
    if version_dir is None or format_type is None:
        print(f"Could not find or determine version directory in: {data_dir}")
        return
    
    # Process based on format type
    if format_type == 'v2':
        _process_lamp_v2_format_files(version_dir, data_dir, output_dir, generated_profile_dir, 
                                     include_generated_profile, k, profile_context_length, 
                                     splits, allowed_splits, task_profile_map)
    else:  # format_type == 'v3+'
        _process_lamp_v3_format_files(version_dir, data_dir, output_dir, generated_profile_dir,
                                     include_generated_profile, k, profile_context_length,
                                     splits, task_profile_map)

def _process_lamp_file(input_file: str, task_name: str, split: str, generated_profile_dir: str, 
                      task_profile_map: Dict, include_generated_profile: bool, 
                      profile_context_length: int, k: List[int], output_dir: str):
    """Process a single LaMP file"""
    
    # Load generated profiles if available
    generated_profiles = {}
    if include_generated_profile:
        # For LaMP, we can either use task-specific files or load all profiles
        if task_name in task_profile_map:
            # Try task-specific file first
            profile_file = os.path.join(generated_profile_dir, task_profile_map[task_name])
            if os.path.exists(profile_file):
                generated_profiles = load_profiles(profile_file)
                print(f"Loaded {len(generated_profiles)} generated profiles for task {task_name} from task-specific file")
        else:
            print(f"No task-specific profile file found for task {task_name}")
            raise ValueError(f"No task-specific profile file found for task {task_name}")
            
        #     else:
        #         # Fallback to loading all profiles
        #         generated_profiles = load_all_generated_profiles(generated_profile_dir)
        #         print(f"Loaded {len(generated_profiles)} generated profiles for task {task_name} from all profile files")
        # else:
        #     # Load all profiles for tasks not in the mapping
        #     generated_profiles = load_all_generated_profiles(generated_profile_dir)
        #     print(f"Loaded {len(generated_profiles)} generated profiles for task {task_name} from all profile files")
    
    data = []
    user_histories = {}  # Store full user histories for augmentation
    
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc=f"Processing LaMP {task_name} {split}"):
            if line.strip():
                item = json.loads(line)
                user_id = item['user_id']
                
                # Store full user data for history augmentation
                user_histories[user_id] = item
                
                # Extract recent profile and generated profile separately
                history = item.get('history', [])
                recent_profile = ""
                if history:
                    recent_profile = extract_recent_user_profile(history, max_context_length=profile_context_length, task_name=task_name)
                
                # Set profile_text to only contain generated profile
                profile_text = ""
                if include_generated_profile and user_id in generated_profiles:
                    profile_text = generated_profiles[user_id]
                
                # Do not consider special tasks: always process only query data
                query = item.get('query', [])
                
                for i, query_item in enumerate(query):
                    question_id = f"lamp_{task_name}_{user_id}_query_{i}"
                    
                    input_text = query_item.get('input', '')
                    output_text = query_item.get('gold', '')
                    
                    # Add BM25 profile retrieval for each k value
                    dataset_item = {
                        'user_id': user_id,
                        'question_id': question_id,
                        'profile_text': profile_text,
                        'profile_all_history': recent_profile,
                        'input': input_text,
                        'output': output_text
                    }

                    # Include history label distribution stats for classification tasks
                    if task_name in ['movie', 'news_cat']:
                        history_stat = _compute_history_label_distribution(task_name, history)
                        dataset_item['history_stat'] = history_stat
                    
                    # Add profile retrieval for each k value using batch retrieval for efficiency
                    if history:
                        retrieval_results = bm25_profile_retrieval_batch(history, input_text, k, task_name=task_name, max_context_length=profile_context_length)
                        for k_val in k:
                            dataset_item[f'profile_retrieval_k{k_val}'] = retrieval_results[k_val]
                    else:
                        for k_val in k:
                            dataset_item[f'profile_retrieval_k{k_val}'] = ""
                    data.append(dataset_item)
    
    # Augment training data with history if it's a small training set
    # Exclude citation and tweet tasks from history enrichment
    if split in ['train', 'ood_train'] and len(data) < 10000 and task_name not in ['citation', 'tweet']:
        print(f"Training dataset has {len(data)} samples, augmenting with history...")
        
        history_samples = []
        # No limit on target samples - process all available history
        print(f"Augmenting with all available history for {len(user_histories)} users")
        # Process each user's history to create additional training samples
        user_list = list(user_histories.keys())
        user_history_tracker = {user_id: len(user_histories[user_id].get('history', [])) - 1 for user_id in user_list}  # Track current history index for each user
        user_idx = 0
        
        while user_idx < len(user_list) * 50:  # Prevent infinite loop, increased limit
            user_id = user_list[user_idx % len(user_list)]
            user_data = user_histories[user_id]
            history = user_data.get('history', [])
            current_history_idx = user_history_tracker[user_id]
            
            if len(history) > 1 and current_history_idx >= 1:  # Need at least 2 history items and valid index
                # Start from most recent history item (highest index) and work backwards
                history_idx = current_history_idx
                
                # Use previous history items as context
                previous_history = history[:history_idx]
                current_history_item = history[history_idx]
                
                # Format history item as training sample
                history_sample = format_lamp_history_as_training_sample(
                    task_name, current_history_item, user_id, history_idx, previous_history, profile_context_length, generated_profiles, include_generated_profile, k
                )
                
                if history_sample:
                    history_samples.append(history_sample)
                
                # Move to previous history item for this user
                user_history_tracker[user_id] -= 1

                
            user_idx += 1
        
        print(f"Generated {len(history_samples)} additional training samples from history")
        data.extend(history_samples)
        
        # Shuffle the combined data to mix original and history samples
        import random
        random.shuffle(data)
        
    elif split in ['train', 'ood_train'] and task_name in ['citation', 'tweet']:
        print(f"Skipping history enrichment for {task_name} task as requested")
    
    if data:
        # Handle both random and ood splits in output path
        # For v3 datasets, use simplified naming (just train/test)
        if split == 'ood_test':
            output_suffix = 'ood_test'
        elif split == 'random_test':
            output_suffix = 'random_test'
        elif split == 'train':
            output_suffix = 'train'
        elif split in ['ood_train', 'ood_test']:
            split_prefix = 'ood'
            split_suffix = split.replace('ood_', '')
            output_suffix = f"{split_prefix}_{split_suffix}"
        else:
            split_prefix = 'random'
            split_suffix = split
            output_suffix = f"{split_prefix}_{split_suffix}"
            
        output_path = os.path.join(output_dir, "LaMP", f"{task_name}_{output_suffix}")
        save_hf_dataset(data, output_path)
        print(f"Saved LaMP {task_name} {split} dataset: {len(data)} samples")

def create_longlamp_dataset(data_dir: str, output_dir: str, generated_profile_dir: str, include_generated_profile: bool = True, k: List[int] = [1], profile_context_length: int = 28000, splits: str = 'both', data_version: str = 'auto'):
    """Convert LongLaMP data to HF format"""
    print("Processing LongLaMP dataset...")
    
    # Map of task names to their profile files
    task_profile_map = {
        'abstract_generation': 'LongLaMP_abstract_generation_profiles.json',
        'product_review': 'LongLaMP_product_review_profiles.json',
        'topic_writing': 'LongLaMP_topic_writing_profiles.json',
    }

    # Determine which splits to include based on the splits parameter
    allowed_splits = []
    if splits == 'random':
        allowed_splits = ['random_train', 'random_test']
    elif splits == 'ood':
        allowed_splits = ['ood_train', 'ood_test']
    else:  # splits == 'both'
        allowed_splits = ['random_train', 'random_test', 'ood_train', 'ood_test']
    
    # Determine which version to use
    version_dir, format_type = determine_version_to_use(data_dir, data_version)
    
    if version_dir is None or format_type is None:
        print(f"Could not find or determine version directory in: {data_dir}")
        return
    
    # Process based on format type
    if format_type == 'v2':
        _process_longlamp_v2_format_files(version_dir, data_dir, output_dir, generated_profile_dir, 
                                         include_generated_profile, k, profile_context_length, 
                                         splits, allowed_splits, task_profile_map)
    else:  # format_type == 'v3+'
        _process_longlamp_v3_format_files(version_dir, data_dir, output_dir, generated_profile_dir,
                                         include_generated_profile, k, profile_context_length,
                                         splits, task_profile_map)

def _process_lamp_v2_format_files(version_dir: str, data_dir: str, output_dir: str, generated_profile_dir: str,
                                  include_generated_profile: bool, k: List[int], profile_context_length: int,
                                  splits: str, allowed_splits: List[str], task_profile_map: Dict):
    """Process LaMP v2 format files"""
    print("Processing LaMP v2 format files...")
    
    # Find all LaMP files in version directory matching the split patterns
    lamp_files = []
    for file in os.listdir(version_dir):
        if (file.endswith('.jsonl') and 
            any(split in file for split in allowed_splits)):
            lamp_files.append(file)
    
    for file_name in lamp_files:
        # Extract task name and split
        # Example: movie_random_test.jsonl -> task_name = movie, split = test
        parts = file_name.replace('.jsonl', '').split('_')
        
        # Find the split (random_train, random_test, ood_train, or ood_test)
        if 'random_train' in file_name:
            split = 'train'
            # Remove 'random' and 'train' from parts to get task name
            task_parts = [p for p in parts if p not in ['random', 'train']]
        elif 'random_test' in file_name:
            split = 'test'
            # Remove 'random' and 'test' from parts to get task name  
            task_parts = [p for p in parts if p not in ['random', 'test']]
        elif 'ood_train' in file_name:
            split = 'ood_train'
            # Remove 'ood' and 'train' from parts to get task name
            task_parts = [p for p in parts if p not in ['ood', 'train']]
        elif 'ood_test' in file_name:
            split = 'ood_test'
            # Remove 'ood' and 'test' from parts to get task name  
            task_parts = [p for p in parts if p not in ['ood', 'test']]
        else:
            continue
            
        task_name = '_'.join(task_parts)
        # Remove "LaMP_processed_" prefix if present
        if task_name.startswith('LaMP_processed_'):
            task_name = task_name[len('LaMP_processed_'):]

        print(f"Processing LaMP task: {task_name}, split: {split}")
        
        input_file = os.path.join(version_dir, file_name)
        
        _process_lamp_file(input_file, task_name, split, generated_profile_dir, task_profile_map,
                          include_generated_profile, profile_context_length, k, output_dir)

def _process_lamp_v3_format_files(version_dir: str, data_dir: str, output_dir: str, generated_profile_dir: str,
                                  include_generated_profile: bool, k: List[int], profile_context_length: int,
                                  splits: str, task_profile_map: Dict):
    """Process LaMP v3+ format files"""
    print("Processing LaMP v3+ format files...")
    
    # Find all LaMP files in version directory
    lamp_files = []
    for file in os.listdir(version_dir):
        if file.endswith('.jsonl'):
            lamp_files.append(file)
    
    for file_name in lamp_files:
        # Extract task name and split for v3 format
        # Example: LaMP_processed_citation_train.jsonl -> task_name = citation, split = train
        
        # Determine split type first
        if file_name.endswith('_train.jsonl'):
            split = 'train'
            # Extract task name: everything between "LaMP_processed_" and "_train.jsonl"
            if file_name.startswith('LaMP_processed_'):
                task_name = file_name[len('LaMP_processed_'):-len('_train.jsonl')]
            else:
                # Fallback: remove _train.jsonl and any prefix
                task_name = file_name.replace('_train.jsonl', '').split('_')[-1]
        elif file_name.endswith('_ood_test.jsonl'):
            split = 'ood_test'
            # Extract task name: everything between "LaMP_processed_" and "_ood_test.jsonl"
            if file_name.startswith('LaMP_processed_'):
                task_name = file_name[len('LaMP_processed_'):-len('_ood_test.jsonl')]
            else:
                # Fallback: remove _ood_test.jsonl and any prefix
                task_name = file_name.replace('_ood_test.jsonl', '').split('_')[-1]
        elif file_name.endswith('_random_test.jsonl'):
            split = 'random_test'
            # Extract task name: everything between "LaMP_processed_" and "_random_test.jsonl"
            if file_name.startswith('LaMP_processed_'):
                task_name = file_name[len('LaMP_processed_'):-len('_random_test.jsonl')]
            else:
                # Fallback: remove _random_test.jsonl and any prefix
                task_name = file_name.replace('_random_test.jsonl', '').split('_')[-1]
        else:
            # Skip files that don't match expected patterns
            continue

        print(f"Processing LaMP v3+ task: {task_name}, split: {split}")
        
        input_file = os.path.join(version_dir, file_name)
        
        _process_lamp_file(input_file, task_name, split, generated_profile_dir, task_profile_map,
                          include_generated_profile, profile_context_length, k, output_dir)

def _process_longlamp_v2_format_files(version_dir: str, data_dir: str, output_dir: str, generated_profile_dir: str,
                                     include_generated_profile: bool, k: List[int], profile_context_length: int,
                                     splits: str, allowed_splits: List[str], task_profile_map: Dict):
    """Process LongLaMP v2 format files"""
    print("Processing LongLaMP v2 format files...")
    
    # Find all LongLaMP files in version directory matching the split patterns
    longlamp_files = []
    for file in os.listdir(version_dir):
        if (file.endswith('.jsonl') and 
            any(split in file for split in allowed_splits)):
            longlamp_files.append(file)
    
    for file_name in longlamp_files:
        # Extract task name and split
        parts = file_name.replace('.jsonl', '').split('_')
        
        # Find the split (random_train, random_test, ood_train, or ood_test)
        if 'random_train' in file_name:
            split = 'train'
            task_parts = [p for p in parts if p not in ['random', 'train']]
        elif 'random_test' in file_name:
            split = 'test'
            task_parts = [p for p in parts if p not in ['random', 'test']]
        elif 'ood_train' in file_name:
            split = 'ood_train'
            task_parts = [p for p in parts if p not in ['ood', 'train']]
        elif 'ood_test' in file_name:
            split = 'ood_test'
            task_parts = [p for p in parts if p not in ['ood', 'test']]
        else:
            continue
            
        task_name = '_'.join(task_parts)
        # Remove "LongLaMP_" prefix if present
        if task_name.startswith('LongLaMP_'):
            task_name = task_name[len('LongLaMP_'):]
        
        print(f"Processing LongLaMP task: {task_name}, split: {split}")
        
        input_file = os.path.join(version_dir, file_name)
        
        _process_longlamp_file(input_file, task_name, split, generated_profile_dir, task_profile_map,
                              include_generated_profile, profile_context_length, k, output_dir)

def _process_longlamp_v3_format_files(version_dir: str, data_dir: str, output_dir: str, generated_profile_dir: str,
                                     include_generated_profile: bool, k: List[int], profile_context_length: int,
                                     splits: str, task_profile_map: Dict):
    """Process LongLaMP v3+ format files"""
    print("Processing LongLaMP v3+ format files...")
    
    # Find all LongLaMP files in version directory
    longlamp_files = []
    for file in os.listdir(version_dir):
        if file.endswith('.jsonl'):
            longlamp_files.append(file)
    
    for file_name in longlamp_files:
        # Extract task name and split for v3 format
        # Example: LongLaMP_abstract_generation_train.jsonl -> task_name = abstract_generation, split = train
        
        # Determine split type first
        if file_name.endswith('_train.jsonl'):
            split = 'train'
            # Extract task name: everything between "LongLaMP_" and "_train.jsonl"
            if file_name.startswith('LongLaMP_'):
                task_name = file_name[len('LongLaMP_'):-len('_train.jsonl')]
            else:
                # Fallback: remove _train.jsonl and any prefix
                task_name = file_name.replace('_train.jsonl', '').split('_')[-1]
        elif file_name.endswith('_ood_test.jsonl'):
            split = 'ood_test'
            # Extract task name: everything between "LongLaMP_" and "_ood_test.jsonl"
            if file_name.startswith('LongLaMP_'):
                task_name = file_name[len('LongLaMP_'):-len('_ood_test.jsonl')]
            else:
                # Fallback: remove _ood_test.jsonl and any prefix
                task_name = file_name.replace('_ood_test.jsonl', '').split('_')[-1]
        elif file_name.endswith('_random_test.jsonl'):
            split = 'random_test'
            # Extract task name: everything between "LongLaMP_" and "_random_test.jsonl"
            if file_name.startswith('LongLaMP_'):
                task_name = file_name[len('LongLaMP_'):-len('_random_test.jsonl')]
            else:
                # Fallback: remove _random_test.jsonl and any prefix
                task_name = file_name.replace('_random_test.jsonl', '').split('_')[-1]
        else:
            # Skip files that don't match expected patterns
            continue

        print(f"Processing LongLaMP v3+ task: {task_name}, split: {split}")
        
        input_file = os.path.join(version_dir, file_name)
        
        _process_longlamp_file(input_file, task_name, split, generated_profile_dir, task_profile_map,
                              include_generated_profile, profile_context_length, k, output_dir)

def _process_longlamp_file(input_file: str, task_name: str, split: str, generated_profile_dir: str, 
                          task_profile_map: Dict, include_generated_profile: bool, 
                          profile_context_length: int, k: List[int], output_dir: str):
    """Process a single LongLaMP file"""
    import time
    
    # Load generated profiles if available
    generated_profiles = {}
    if include_generated_profile:
        # For LongLaMP, we use task-specific files
        if task_name in task_profile_map:
            # Try task-specific file first
            profile_file = os.path.join(generated_profile_dir, task_profile_map[task_name])
            if os.path.exists(profile_file):
                generated_profiles = load_profiles(profile_file)
                print(f"Loaded {len(generated_profiles)} generated profiles for task {task_name} from task-specific file")
        else:
            print(f"No task-specific profile file found for task {task_name}")
            raise ValueError(f"No task-specific profile file found for task {task_name}")
    
    # Get file size for debugging
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # Size in MB
    print(f"File size: {file_size:.2f} MB")
    
    # First, count total lines for accurate progress bar
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
        print(f"Total lines to process: {total_lines}")
    
    data = []
    user_histories = {}  # Store full user histories for augmentation
    
    with open(input_file, 'r') as f:
        # Create progress bar with more detailed info
        pbar = tqdm(f, total=total_lines, desc=f"Processing LongLaMP {task_name} {split}")
        
        for line_idx, line in enumerate(pbar):
            if line.strip():
                try:
                    start_time = time.time()
                    
                    item = json.loads(line)
                    user_id = item['user_id']
                    
                    # Store full user data for history augmentation
                    user_histories[user_id] = item
                    
                    # Update progress bar with current user
                    pbar.set_postfix({'user': user_id, 'line': line_idx + 1})
                    
                    # Extract history with size info
                    history = item.get('history', [])
                    history_size = len(str(history)) / 1024  # Size in KB
                    
                    if history_size > 100:  # Log if history is larger than 100KB
                        print(f"\nLarge history for user {user_id}: {history_size:.2f} KB, {len(history)} items")
                    
                    # Extract recent profile and generated profile separately
                    recent_profile = ""
                    
                    # Extract recent user profile from history if available
                    if history:
                        profile_start = time.time()
                        recent_profile = extract_recent_user_profile(history, max_context_length=profile_context_length, task_name=task_name)
                        profile_time = time.time() - profile_start
                        
                        if profile_time > 1.0:  # Log if profile extraction takes more than 1 second
                            print(f"\nSlow profile extraction for user {user_id}: {profile_time:.2f}s")
                    
                    # Set profile_text to only contain generated profile
                    profile_text = ""
                    if include_generated_profile and user_id in generated_profiles:
                        profile_text = generated_profiles[user_id]
                    
                    # Only consider query data for LongLaMP (unified_single_profile_mode)
                    query = item.get('query', [])
                    
                    for i, query_item in enumerate(query):
                        question_id = f"longlamp_{task_name}_{user_id}_query_{i}"
                        
                        input_text = query_item.get('input', '')
                        # Handle both 'gold' and 'output' fields
                        output_text = query_item.get('gold', '') or query_item.get('output', '')
                        
                        # Add BM25 profile retrieval for each k value
                        dataset_item = {
                            'user_id': user_id,
                            'question_id': question_id,
                            'profile_text': profile_text,
                            'profile_all_history': recent_profile,
                            'input': input_text,
                            'output': output_text
                        }
                        
                        # Add profile retrieval for each k value using batch retrieval for efficiency
                        if history:
                            retrieval_results = bm25_profile_retrieval_batch(history, input_text, k, task_name=task_name, max_context_length=profile_context_length)
                            for k_val in k:
                                dataset_item[f'profile_retrieval_k{k_val}'] = retrieval_results[k_val]
                        else:
                            for k_val in k:
                                dataset_item[f'profile_retrieval_k{k_val}'] = ""
                        data.append(dataset_item)
                    
                    # Log processing time for slow items
                    process_time = time.time() - start_time
                    if process_time > 2.0:
                        print(f"\nSlow processing for user {user_id}: {process_time:.2f}s")
                        
                except json.JSONDecodeError as e:
                    print(f"\nError parsing JSON at line {line_idx + 1}: {e}")
                    continue
                except Exception as e:
                    print(f"\nError processing user {user_id} at line {line_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Augment training data with history if it's a small training set
    if split in ['train', 'ood_train'] and len(data) < 10000:
        print(f"Training dataset has {len(data)} samples, augmenting with history...")
        
        history_samples = []
        # No limit on target samples - process all available history
        
        # Process each user's history to create additional training samples
        user_list = list(user_histories.keys())
        user_history_tracker = {user_id: len(user_histories[user_id].get('history', [])) - 1 for user_id in user_list}  # Track current history index for each user
        user_idx = 0
        
        while user_idx < len(user_list) * 50:  # Prevent infinite loop, increased limit
            user_id = user_list[user_idx % len(user_list)]
            user_data = user_histories[user_id]
            history = user_data.get('history', [])
            current_history_idx = user_history_tracker[user_id]
            
            if len(history) > 1 and current_history_idx >= 1:  # Need at least 2 history items and valid index
                # Start from most recent history item (highest index) and work backwards
                history_idx = current_history_idx
                
                # Use previous history items as context
                previous_history = history[:history_idx]
                current_history_item = history[history_idx]
                
                # Format history item as training sample
                history_sample = format_longlamp_history_as_training_sample(
                    task_name, current_history_item, user_id, history_idx, previous_history, profile_context_length, generated_profiles, include_generated_profile, k
                )
                
                if history_sample:
                    history_samples.append(history_sample)
                
                # Move to previous history item for this user
                user_history_tracker[user_id] -= 1
            
            user_idx += 1
        
        print(f"Generated {len(history_samples)} additional training samples from history")
        data.extend(history_samples)
        
        # Shuffle the combined data to mix original and history samples
        import random
        random.shuffle(data)
    
    if data:
        # Handle both random and ood splits in output path
        # For v3 datasets, use simplified naming (just train/test)
        if split == 'ood_test':
            output_suffix = 'ood_test'
        elif split == 'random_test':
            output_suffix = 'random_test'
        elif split == 'train':
            output_suffix = 'train'
        elif split in ['ood_train', 'ood_test']:
            split_prefix = 'ood'
            split_suffix = split.replace('ood_', '')
            output_suffix = f"{split_prefix}_{split_suffix}"
        else:
            split_prefix = 'random'
            split_suffix = split
            output_suffix = f"{split_prefix}_{split_suffix}"
            
        output_path = os.path.join(output_dir, "LongLaMP", f"{task_name}_{output_suffix}")
        save_hf_dataset(data, output_path)
        print(f"Saved LongLaMP {task_name} {split} dataset: {len(data)} samples")

def create_prism_dataset(data_dir: str, output_dir: str, splits: str = 'both', data_version: str = 'auto', score_threshold: int = 80):
    """Convert PRISM data to HF format with combined train and separate test datasets"""
    print("Processing PRISM dataset...")
    
    # Determine which version to use
    version_dir, format_type = determine_version_to_use(data_dir, data_version)
    
    if version_dir is None or format_type is None:
        print(f"Could not find or determine version directory in: {data_dir}")
        return
    
    import glob
    
    # Collect all train and test data
    train_data = []
    random_test_data = []
    ood_test_data = []
    
    # Determine which splits to load based on the splits parameter
    load_random = splits in ['random', 'both']
    load_ood = splits in ['ood', 'both']
    
    # Process train splits
    train_splits = []
    if load_random:
        train_splits.append('random_train')
    if load_ood:
        train_splits.append('ood_train')
        
    for split in train_splits:
        pattern = os.path.join(version_dir, f"*_{split}.jsonl")
        matching_files = glob.glob(pattern)
        
        # If no specific train split found, try generic train pattern
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            # Try fallback to generic train pattern
            generic_pattern = os.path.join(version_dir, "*_train.jsonl")
            matching_files = glob.glob(generic_pattern)
            if matching_files:
                print(f"Found generic train file: {matching_files[0]}")
            else:
                print(f"No train files found at all")
                continue
        
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")
        
        # Track filtering statistics
        total_samples = 0
        filtered_samples = 0  # includes score + model_name filtering
        
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc=f"Processing PRISM {split}"):
                if line.strip():
                    item = json.loads(line)
                    user_id = item['user_id']
                    
                    # Get profile text - only original profile, no conversation history
                    profile_text = ""
                    
                    # Add original profile if available
                    if 'profile_text' in item:
                        profile_text = _to_plain_text(item.get('profile_text', ""))
                    
                    # No recent profile for PRISM data
                    recent_profile = ""
                    
                    # Get conversations for score checking
                    conversations = item.get('conversations', [])
                    
                    # Process query data
                    query = item.get('query', [])
                    
                    for i, query_item in enumerate(query):
                        total_samples += 1
                        
                        # Apply filtering for training data only (score + model_name contains 'gpt-4')
                        if 'train' in split:  # This is a training split
                            # Find the corresponding conversation and check the chosen response
                            conversation_idx = query_item.get('conversation_idx', i)
                            if conversation_idx < len(conversations):
                                conversation = conversations[conversation_idx]
                                conversation_history = conversation.get('conversation_history', [])

                                # Find the chosen response info from the conversation history
                                # The output corresponds to the chosen response from the conversation
                                chosen_score = None
                                chosen_model_name = None

                                # Determine expected output content
                                query_output_content = query_item.get('output', {})
                                if isinstance(query_output_content, dict):
                                    expected_output = query_output_content.get('content', '')
                                else:
                                    expected_output = str(query_output_content)

                                # Find the turn with matching content and if_chosen=True
                                for turn in conversation_history:
                                    if (
                                        turn.get('if_chosen', False)
                                        and turn.get('role') == 'model'
                                        and turn.get('content', '') == expected_output
                                    ):
                                        chosen_score = turn.get('score')
                                        chosen_model_name = turn.get('model_name')
                                        break

                                # If we didn't find an exact match, fall back to just the chosen response
                                if chosen_score is None and chosen_model_name is None:
                                    for turn in conversation_history:
                                        if turn.get('if_chosen', False) and turn.get('role') == 'model':
                                            chosen_score = turn.get('score')
                                            chosen_model_name = turn.get('model_name')
                                            break

                                # Score filter
                                if chosen_score is not None and chosen_score < score_threshold:
                                    filtered_samples += 1
                                    continue

                                # model_name filter: require containing 'gpt-4'
                                model_name_str = str(chosen_model_name or '')
                                if 'gpt-4' not in model_name_str.lower():
                                    filtered_samples += 1
                                    continue
                            else:
                                # Cannot validate chosen model_name without conversation; skip
                                filtered_samples += 1
                                continue
                        
                        question_id = f"prism_{user_id}_query_{i}_{split}"
                        
                        # Extract input from the input list (get the content from the last message)
                        input_messages = query_item.get('input', [])
                    
                        
                        # Extract output content from the output dict
                        output_data = query_item.get('output', {})
                        if isinstance(output_data, dict):
                            output_text = output_data.get('content', '')
                        else:
                            output_text = str(output_data)
                        
                        dataset_item = {
                            'user_id': user_id,
                            'question_id': question_id,
                            'profile_text': profile_text,
                            'profile_all_history': recent_profile,
                            'input': input_messages,
                            'output': output_text
                        }
                        train_data.append(dataset_item)
        
        # Print filtering statistics for training splits
        if 'train' in split and total_samples > 0:
            kept_samples = total_samples - filtered_samples
            print(
                f"Filtering for {split}: kept {kept_samples}/{total_samples} samples "
                f"({kept_samples/total_samples*100:.1f}%) with score >= {score_threshold} "
                f"and model_name contains 'gpt-4'"
            )
    
    # Process test splits
    test_splits = []
    if load_random:
        test_splits.append('random_test')
    if load_ood:
        test_splits.append('ood_test')
        
    for split in test_splits:
        pattern = os.path.join(version_dir, f"*_{split}.jsonl")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            continue
        
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")
        
        # Track filtering statistics for test splits as well
        total_samples = 0
        filtered_samples = 0  # includes score + model_name filtering

        current_test_data = []
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc=f"Processing PRISM {split}"):
                if line.strip():
                    item = json.loads(line)
                    user_id = item['user_id']
                    
                    # Get profile text - only original profile, no conversation history
                    profile_text = ""
                    
                    # Add original profile if available
                    if 'profile_text' in item:
                        profile_text = _to_plain_text(item.get('profile_text', ""))
                    
                    # No recent profile for PRISM data
                    recent_profile = ""
                    
                    # Get conversations for score checking
                    conversations = item.get('conversations', [])

                    # Process query data
                    query = item.get('query', [])
                    
                    for i, query_item in enumerate(query):
                        total_samples += 1

                        # Apply filtering for test data as requested (score + model_name contains 'gpt-4')
                        # Find the corresponding conversation and check the chosen response
                        conversation_idx = query_item.get('conversation_idx', i)
                        if conversation_idx < len(conversations):
                            conversation = conversations[conversation_idx]
                            conversation_history = conversation.get('conversation_history', [])

                            # Find the chosen response info from the conversation history
                            chosen_score = None
                            chosen_model_name = None

                            # Determine expected output content
                            query_output_content = query_item.get('output', {})
                            if isinstance(query_output_content, dict):
                                expected_output = query_output_content.get('content', '')
                            else:
                                expected_output = str(query_output_content)

                            # Find the turn with matching content and if_chosen=True
                            for turn in conversation_history:
                                if (
                                    turn.get('if_chosen', False)
                                    and turn.get('role') == 'model'
                                    and turn.get('content', '') == expected_output
                                ):
                                    chosen_score = turn.get('score')
                                    chosen_model_name = turn.get('model_name')
                                    break

                            # If we didn't find an exact match, fall back to just the chosen response
                            if chosen_score is None and chosen_model_name is None:
                                for turn in conversation_history:
                                    if turn.get('if_chosen', False) and turn.get('role') == 'model':
                                        chosen_score = turn.get('score')
                                        chosen_model_name = turn.get('model_name')
                                        break

                            # Skip this sample if chosen response score < threshold
                            if chosen_score is not None and chosen_score < score_threshold:
                                filtered_samples += 1
                                continue

                            # model_name filter: require containing 'gpt-4'
                            model_name_str = str(chosen_model_name or '')
                            if 'gpt-4' not in model_name_str.lower():
                                filtered_samples += 1
                                continue
                        else:
                            # Cannot validate chosen model_name without conversation; skip
                            filtered_samples += 1
                            continue

                        question_id = f"prism_{user_id}_query_{i}_{split}"
                        
                        # Extract input from the input list (get the content from the last message)
                        input_messages = query_item.get('input', [])
                    
                        
                        # Extract output content from the output dict
                        output_data = query_item.get('output', {})
                        if isinstance(output_data, dict):
                            output_text = output_data.get('content', '')
                        else:
                            output_text = str(output_data)
                        
                        dataset_item = {
                            'user_id': user_id,
                            'question_id': question_id,
                            'profile_text': profile_text,
                            'profile_all_history': recent_profile,
                            'input': input_messages,
                            'output': output_text
                        }
                        current_test_data.append(dataset_item)
        
        # Print filtering statistics for test splits
        if total_samples > 0:
            kept_samples = total_samples - filtered_samples
            print(
                f"Filtering for {split}: kept {kept_samples}/{total_samples} samples "
                f"({kept_samples/total_samples*100:.1f}%) with score >= {score_threshold} "
                f"and model_name contains 'gpt-4'"
            )
        
        # Store test data in appropriate list
        if split == 'random_test':
            random_test_data = current_test_data
        elif split == 'ood_test':
            ood_test_data = current_test_data
    
    # Deduplicate train data against test datasets
    test_datasets = [random_test_data, ood_test_data]
    deduplicated_train_data = deduplicate_train_data(train_data, test_datasets)
    
    # Save datasets
    if deduplicated_train_data:
        output_path = os.path.join(output_dir, "PRISM", "train")
        save_hf_dataset(deduplicated_train_data, output_path)
        print(f"Saved PRISM train dataset: {len(deduplicated_train_data)} samples")
    
    if random_test_data:
        output_path = os.path.join(output_dir, "PRISM", "random_test")
        save_hf_dataset(random_test_data, output_path)
        print(f"Saved PRISM random_test dataset: {len(random_test_data)} samples")
    
    if ood_test_data:
        output_path = os.path.join(output_dir, "PRISM", "ood_test")
        save_hf_dataset(ood_test_data, output_path)
        print(f"Saved PRISM ood_test dataset: {len(ood_test_data)} samples")

def create_aloe_dataset(data_dir: str, output_dir: str, splits: str = 'both', data_version: str = 'auto'):
    """Convert ALOE data to HF format similar to PRISM.

    - Input: message list of dicts with keys {'role', 'content'}
    - Output: string (the preferred assistant response for the last turn)
    - Each turn in a conversation becomes a data point
    - For previous turns, use the assistant content from the 'chosen' entry
    """
    print("Processing ALOE dataset...")

    # Determine which version to use
    version_dir, _ = determine_version_to_use(data_dir, data_version)

    if version_dir is None:
        print(f"Could not find or determine version directory in: {data_dir}")
        return

    import glob

    train_data = []
    random_test_data = []
    ood_test_data = []

    # Train split: typically only one file ALOE_train.jsonl
    train_pattern = os.path.join(version_dir, "ALOE_train.jsonl")
    matching_train = glob.glob(train_pattern)
    if not matching_train:
        # Fallback to generic pattern
        matching_train = glob.glob(os.path.join(version_dir, "*_train.jsonl"))
    if matching_train:
        input_file = matching_train[0]
        print(f"Processing file: {input_file}")
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc=f"Processing ALOE train"):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                user_id = item.get('user_id')
                profile_text = _to_plain_text(item.get('profile_text', ""))

                conversations = item.get('conversations', [])
                for conv_idx, conv in enumerate(conversations):
                    conv_id = conv.get('conversation_id', f"c{conv_idx}")
                    history = conv.get('conversation_history', [])
                    # Each element of history contains: 'user', 'assistant': {'preferred','rejected'}, 'chosen'
                    for turn_idx in range(len(history)):
                        # Build input messages: previous user-assistant (assistant uses chosen), then current user
                        input_messages = []
                        try:
                            for j in range(turn_idx):
                                prev = history[j]
                                user_msg = prev.get('user')
                                if user_msg:
                                    input_messages.append({'role': 'user', 'content': user_msg})
                                # assistant message based on chosen entry
                                chosen_key = prev.get('chosen', 'preferred')
                                assistant_obj = prev.get('assistant', {})
                                assistant_msg = None
                                if isinstance(assistant_obj, dict):
                                    assistant_msg = assistant_obj.get(chosen_key) or assistant_obj.get('preferred') or assistant_obj.get('rejected')
                                if assistant_msg:
                                    input_messages.append({'role': 'assistant', 'content': assistant_msg})

                            # Add current user message (last message)
                            curr = history[turn_idx]
                            curr_user_msg = curr.get('user', '')
                            if not curr_user_msg:
                                # Skip if no user message
                                continue
                            input_messages.append({'role': 'user', 'content': curr_user_msg})

                            # Output is always the preferred assistant reply for the current turn
                            assistant_obj = curr.get('assistant', {})
                            output_text = ""
                            if isinstance(assistant_obj, dict):
                                output_text = assistant_obj.get('preferred', '')
                            else:
                                # Unexpected shape; skip
                                continue
                            if not output_text:
                                continue

                            question_id = f"aloe_{user_id}_{conv_id}_t{turn_idx}_train"
                            dataset_item = {
                                'user_id': user_id,
                                'question_id': question_id,
                                'profile_text': profile_text,
                                'profile_all_history': "",
                                'input': input_messages,
                                'output': output_text
                            }
                            train_data.append(dataset_item)
                        except Exception:
                            continue

    # Test splits
    test_specs = []
    if splits in ['random', 'both']:
        test_specs.append(('random_test', os.path.join(version_dir, "ALOE_random_test.jsonl")))
    if splits in ['ood', 'both']:
        test_specs.append(('ood_test', os.path.join(version_dir, "ALOE_ood_test.jsonl")))

    for split_name, pattern in test_specs:
        matching_files = glob.glob(pattern) or glob.glob(os.path.join(version_dir, f"*_{split_name}.jsonl"))
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            continue
        input_file = matching_files[0]
        print(f"Processing file: {input_file}")

        current_test_data = []
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc=f"Processing ALOE {split_name}"):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                user_id = item.get('user_id')
                profile_text = _to_plain_text(item.get('profile_text', ""))

                conversations = item.get('conversations', [])
                for conv_idx, conv in enumerate(conversations):
                    conv_id = conv.get('conversation_id', f"c{conv_idx}")
                    history = conv.get('conversation_history', [])
                    for turn_idx in range(len(history)):
                        input_messages = []
                        try:
                            for j in range(turn_idx):
                                prev = history[j]
                                user_msg = prev.get('user')
                                if user_msg:
                                    input_messages.append({'role': 'user', 'content': user_msg})
                                chosen_key = prev.get('chosen', 'preferred')
                                assistant_obj = prev.get('assistant', {})
                                assistant_msg = None
                                if isinstance(assistant_obj, dict):
                                    assistant_msg = assistant_obj.get(chosen_key) or assistant_obj.get('preferred') or assistant_obj.get('rejected')
                                if assistant_msg:
                                    input_messages.append({'role': 'assistant', 'content': assistant_msg})

                            curr = history[turn_idx]
                            curr_user_msg = curr.get('user', '')
                            if not curr_user_msg:
                                continue
                            input_messages.append({'role': 'user', 'content': curr_user_msg})

                            assistant_obj = curr.get('assistant', {})
                            output_text = ""
                            if isinstance(assistant_obj, dict):
                                output_text = assistant_obj.get('preferred', '')
                            if not output_text:
                                continue

                            question_id = f"aloe_{user_id}_{conv_id}_t{turn_idx}_{split_name}"
                            dataset_item = {
                                'user_id': user_id,
                                'question_id': question_id,
                                'profile_text': profile_text,
                                'profile_all_history': "",
                                'input': input_messages,
                                'output': output_text
                            }
                            current_test_data.append(dataset_item)
                        except Exception:
                            continue

        if split_name == 'random_test':
            random_test_data = current_test_data
        elif split_name == 'ood_test':
            ood_test_data = current_test_data

    # Deduplicate train data against test datasets
    test_datasets = [random_test_data, ood_test_data]
    deduplicated_train_data = deduplicate_train_data(train_data, test_datasets)

    # Save datasets
    if deduplicated_train_data:
        output_path = os.path.join(output_dir, "ALOE", "train")
        save_hf_dataset(deduplicated_train_data, output_path)
        print(f"Saved ALOE train dataset: {len(deduplicated_train_data)} samples")

    if random_test_data:
        output_path = os.path.join(output_dir, "ALOE", "random_test")
        save_hf_dataset(random_test_data, output_path)
        print(f"Saved ALOE random_test dataset: {len(random_test_data)} samples")

    if ood_test_data:
        output_path = os.path.join(output_dir, "ALOE", "ood_test")
        save_hf_dataset(ood_test_data, output_path)
        print(f"Saved ALOE ood_test dataset: {len(ood_test_data)} samples")

def main():
    parser = argparse.ArgumentParser(description="Convert all datasets to HF format")
    parser.add_argument("--data_dir", type=str, default=".",
                       help="Root directory containing all dataset folders")
    parser.add_argument("--output_dir", type=str, default="hf_datasets",
                       help="Output directory for HF datasets")
    parser.add_argument("--include_generated_profile", action="store_true", default=False,
                       help="Whether to include generated profile in LaMP and LongLaMP datasets (default: False)")
    parser.add_argument("--generated_profile_dir", type=str, default=None,
                       help="Directory containing generated profile JSON files (default: {data_dir}/generated_profile)")
    parser.add_argument("--k", type=int, nargs='+', default=[1,2,4],
                       help="List of k values for BM25 profile retrieval")
    parser.add_argument("--profile_context_length", type=int, default=23000,
                       help="Maximum token count for user profile text (default: 28000)")
    parser.add_argument("--splits", type=str, choices=['random', 'ood', 'both'], default='both',
                       help="Which splits to process: 'random' for random splits only, 'ood' for OOD splits only, or 'both' for all splits (default: both)")
    parser.add_argument("--data_version", type=str, default='auto',
                       help="Which data version to use: 'v2' for v2 format, 'v3' for v3 format, 'v4', 'v5', etc. for higher versions (use v3 format), or 'auto' to automatically prioritize highest version (default: auto)")
    parser.add_argument("--score_threshold", type=int, default=80,
                       help="Minimum score threshold for PRISM training data filtering. Only responses with score >= threshold will be kept (default: 80)")
    
    args = parser.parse_args()
    
    # Validate data_version argument
    if args.data_version != 'auto' and not (args.data_version.startswith('v') and args.data_version[1:].isdigit()):
        print(f"Error: Invalid data_version '{args.data_version}'. Must be 'auto' or version format like 'v2', 'v3', 'v4', etc.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataset directories
    ec_dir = os.path.join(args.data_dir, "EC")
    opinionqa_dir = os.path.join(args.data_dir, "OpinionQA")
    personalreddit_dir = os.path.join(args.data_dir, "PersonalReddit")
    lamp_dir = os.path.join(args.data_dir, "LaMP")
    longlamp_dir = os.path.join(args.data_dir, "LongLaMP")
    prism_dir = os.path.join(args.data_dir, "PRISM")
    aloe_dir = os.path.join(args.data_dir, "ALOE")
    
    # Set generated profile directory - use provided path or default
    if args.generated_profile_dir:
        generated_profile_dir = args.generated_profile_dir
    else:
        generated_profile_dir = os.path.join(args.data_dir, "generated_profile")
    
    # Process each dataset
    # if os.path.exists(lamp_dir):
    #     create_lamp_dataset(lamp_dir, args.output_dir, generated_profile_dir, include_generated_profile=args.include_generated_profile, k=args.k, profile_context_length=args.profile_context_length, splits=args.splits, data_version=args.data_version)
    
    if os.path.exists(ec_dir):
        create_ec_dataset(ec_dir, args.output_dir, splits=args.splits, data_version=args.data_version)
    
    if os.path.exists(opinionqa_dir):
        create_opinionqa_dataset(opinionqa_dir, args.output_dir, splits=args.splits, data_version=args.data_version)
    
    if os.path.exists(personalreddit_dir):
        create_personalreddit_dataset(personalreddit_dir, args.output_dir, splits=args.splits, data_version=args.data_version)

    # if os.path.exists(longlamp_dir):
    #     create_longlamp_dataset(longlamp_dir, args.output_dir, generated_profile_dir, include_generated_profile=args.include_generated_profile, k=args.k, profile_context_length=args.profile_context_length, splits=args.splits, data_version=args.data_version)
    
    if os.path.exists(prism_dir):
        create_prism_dataset(prism_dir, args.output_dir, splits=args.splits, data_version=args.data_version, score_threshold=args.score_threshold)

    # if os.path.exists(aloe_dir):
    #     create_aloe_dataset(aloe_dir, args.output_dir, splits=args.splits, data_version=args.data_version)
    
    print("\nAll datasets have been converted to Hugging Face format!")
    print(f"Output directory: {args.output_dir}")
    
    if args.include_generated_profile:
        print(f"\nGenerated profiles were included from: {generated_profile_dir}")
        print("Note: Generated profiles are automatically appended to LaMP and LongLaMP datasets during creation.")
    
    print("\nTo load these datasets, you can use:")
    print("from datasets import load_from_disk")
    print("dataset = load_from_disk('path/to/dataset')")

if __name__ == "__main__":
    main()
