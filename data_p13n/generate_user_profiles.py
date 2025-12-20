import asyncio
import json
import os
import argparse
from pathlib import Path
from openai import AsyncOpenAI
from jinja2 import Template
import random
from tqdm import tqdm

def estimate_token_count(text):
    """
    Rough estimation of token count for a given text.
    Uses the approximation that 1 token ≈ 4 characters for English text.
    """
    return len(text) // 4

def format_history_item_truncated(item, detected_keys=None, max_value_length=300):
    """Format a single history item including all keys, with value truncation for long content"""
    if not isinstance(item, dict):
        return ""
    
    # Format each history item including all available keys
    entry_parts = []
    
    # Process ALL keys in the item, not just detected ones
    # Sort keys for consistent ordering, with detected_keys first if provided
    all_keys = list(item.keys())
    if detected_keys:
        # Put detected keys first (most common/important), then remaining keys
        ordered_keys = detected_keys + [k for k in all_keys if k not in detected_keys]
    else:
        # Use all keys in alphabetical order for consistency
        ordered_keys = sorted(all_keys)
    
    for key in ordered_keys:
        if key in item and item[key] is not None:
            value = str(item[key]).strip()
            if value:  # Only add non-empty values
                # Truncate long values to keep profiles concise
                if len(value) > max_value_length:
                    value = value[:max_value_length] + "..."
                # Capitalize the key for display
                display_key = key.replace('_', ' ').title()
                entry_parts.append(f"{display_key}: {value}")
    
    if entry_parts:
        return "\n".join(entry_parts)  # Use newline separator for readability
    else:
        return f"User activity: {str(item)[:200]}"

def truncate_history_to_fit_context(history_items, prompt_template, max_context_length, file_name=""):
    """
    Truncate history values within items to fit within context length.
    First try truncating individual values, then remove items from beginning if still too long.
    Returns the processed history that fits within the context limit.
    """
    if not history_items or max_context_length <= 0:
        return history_items
    
    # Detect the format of history items
    detected_keys = detect_history_format(history_items)
    
    # First, try truncating values within items with progressively smaller limits
    for max_value_length in [500]:
        # Create truncated versions of all history items
        truncated_items = []
        for item in history_items:
            if isinstance(item, dict):
                truncated_item = {}
                for key, value in item.items():
                    if value is not None:
                        value_str = str(value).strip()
                        if len(value_str) > max_value_length:
                            truncated_item[key] = value_str[:max_value_length] + "..."
                        else:
                            truncated_item[key] = value
                    else:
                        truncated_item[key] = value
                truncated_items.append(truncated_item)
            else:
                truncated_items.append(item)
        
        # Check if this fits within context
        user_history_text = history_to_string(truncated_items, file_name, detected_keys)
        prompt_text = prompt_template.render(user_history=user_history_text)
        estimated_tokens = estimate_token_count(prompt_text)
        
        # Add some buffer (10%) to be safe
        if estimated_tokens <= max_context_length * 0.9:
            return truncated_items
    
    # If truncating values isn't enough, progressively remove items from beginning
    for start_idx in range(len(history_items)):
        truncated_history = history_items[start_idx:]
        
        # Apply value truncation to remaining items
        truncated_items = []
        for item in truncated_history:
            if isinstance(item, dict):
                truncated_item = {}
                for key, value in item.items():
                    if value is not None:
                        value_str = str(value).strip()
                        if len(value_str) > 300:  # Use minimal truncation
                            truncated_item[key] = value_str[:300] + "..."
                        else:
                            truncated_item[key] = value
                    else:
                        truncated_item[key] = value
                truncated_items.append(truncated_item)
            else:
                truncated_items.append(item)
        
        # Generate the history text and prompt to estimate tokens
        user_history_text = history_to_string(truncated_items, file_name, detected_keys)
        prompt_text = prompt_template.render(user_history=user_history_text)
        
        # Estimate token count
        estimated_tokens = estimate_token_count(prompt_text)
        
        # Add some buffer (10%) to be safe
        if estimated_tokens <= max_context_length * 0.9:
            return truncated_items
    
    # If even a single history item is too large, return empty list
    print(f"Warning: Even single history item exceeds context limit. Using empty history.")
    return []

async def call_llm_async(client, messages, results, custom_id, model_name, temperature=1.0, max_tokens=10000, top_p=0.95, frequency_penalty=0, presence_penalty=0, stop=None, task_name=None):
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
        
        choice = response.choices[0]
        if choice.finish_reason not in ['stop', 'length']:
            if 'content_filter' in choice.finish_reason:
                content = "Error: content filtered due to OpenAI policy."
            else:
                raise ValueError(f"OpenAI Finish Reason Error: {choice.finish_reason}")
        else:
            content = choice.message.content.strip()
        
        result = {"custom_id": custom_id, "content": content}
        if task_name is not None:
            result["task_name"] = task_name
        results.append(result)
        
    except Exception as e:
        print(f"Task {custom_id} failed: {e}. Retrying in 5 seconds...")
        await asyncio.sleep(random.randint(5, 15))
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            
            choice = response.choices[0]
            if choice.finish_reason not in ['stop', 'length']:
                if 'content_filter' in choice.finish_reason:
                    content = "Error: content filtered due to OpenAI policy."
                else:
                    raise ValueError(f"OpenAI Finish Reason Error: {choice.finish_reason}")
            else:
                content = choice.message.content.strip()
            
            result = {"custom_id": custom_id, "content": content}
            if task_name is not None:
                result["task_name"] = task_name
            results.append(result)
            
        except Exception as e:
            print(f"Task {custom_id} failed again: {e}. Skipping...")
            result = {"custom_id": custom_id, "content": f"Error: {str(e)}"}
            if task_name is not None:
                result["task_name"] = task_name
            results.append(result)
    finally:
        await asyncio.sleep(random.uniform(0.1, 1.0))

async def call_llm_in_parallel(client, requests, model_name, output_path=None, batch_size=10, temperature=1.0, max_tokens=10000, top_p=0.95, frequency_penalty=0, presence_penalty=0, stop=None):
    """
    Modified: output_path is now a dict mapping from task name (input file stem) to output file path.
    Each request must have a 'task_name' key to determine which file to write to.
    This function writes checkpoint output files after each batch is processed.
    """
    num_calls = len(requests)
    results = []
    batch_results = []
    print("Starting LLM calls...")
    
    for i in range(0, num_calls, batch_size):
        batch_tasks = []
        for j in range(batch_size):
            if i + j < num_calls:
                _request = requests[i + j]
                _messages = _request['body']['messages']
                _custom_id = _request['custom_id']
                
                # Use parameters from request body if available, otherwise use function parameters
                _model_name = _request['body'].get('model', model_name)
                _temperature = _request['body'].get('temperature', temperature)
                _max_tokens = _request['body'].get('max_tokens', max_tokens)
                _top_p = _request['body'].get('top_p', top_p)
                _frequency_penalty = _request['body'].get('frequency_penalty', frequency_penalty)
                _presence_penalty = _request['body'].get('presence_penalty', presence_penalty)
                _stop = _request['body'].get('stop', stop)
                _task_name = _request.get('task_name', None)
                
                batch_tasks.append(call_llm_async(
                    client, _messages, batch_results, _custom_id, _model_name, 
                    _temperature, _max_tokens, _top_p, _frequency_penalty, 
                    _presence_penalty, _stop, task_name=_task_name
                ))
        
        await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Write results to the correct output file for each task (checkpointing after each batch)
        if output_path:
            # output_path is a dict: {task_name: output_file_path}
            # Each result in batch_results must have a 'task_name' key
            task_results = {}
            for result in batch_results:
                task_name = result.get('task_name')
                if task_name is None:
                    # Fallback: if not present, write to a default file if exists
                    task_name = "default"
                if task_name not in task_results:
                    task_results[task_name] = []
                task_results[task_name].append(result)
            for task_name, results_list in task_results.items():
                file_path = output_path.get(task_name)
                if file_path is None:
                    continue
                # Write checkpoint output file for this batch
                with open(file_path, 'a') as f:
                    for result in results_list:
                        # Remove 'task_name' before writing
                        result_to_write = dict(result)
                        result_to_write.pop('task_name', None)
                        f.write(json.dumps(result_to_write) + '\n')
            print(f"Batch {i // batch_size + 1} completed and saved to files. Total tasks completed: {i + len(batch_tasks)} / {num_calls}", flush=True)
        else:
            print(f"Batch {i // batch_size + 1} completed. Total tasks completed: {i + len(batch_tasks)} / {num_calls}", flush=True)
        
        batch_results.clear()
    
    print("All tasks completed.")
    return results


def load_prompt_template(file_path):
    """Load the prompt template from file"""
    with open(file_path, 'r') as f:
        return Template(f.read())


task2discription = {
    'citation': 'Academic citation recommendation: Identify relevant reference papers for researchers based on their publication titles and research focus areas.',
    'movie': 'Movie genre tagging: Analyze movie descriptions and assign appropriate genre tags based on content themes, narrative elements, and stylistic features.',
    'news_cat': 'News article categorization: Classify news articles into topical categories based on content, subject matter, and thematic focus.',
    'news_headline': 'News article headline generation: Create a concise and engaging headline for news articles based on their content and key themes.',
    'product': 'Product review rating prediction: Analyze product reviews and predict the rating score based on sentiment, content quality, and expressed satisfaction levels.',
    'scholarly_title': 'Academic title generation: Generate concise and descriptive titles for research papers based on abstracts, capturing the main research contribution and scope.',
    'tweet': 'Tweet paraphrasing: Rewrite tweets in a personal style while maintaining the original meaning and adapting the tone and language to individual communication patterns.',
    'topic_writing': 'Topic-based content generation: Create personalized Reddit posts on given topics that reflect individual writing style, interests, and communication preferences.',
    'product_review': 'Product review generation: Write detailed product reviews that reflect personal experiences, preferences, and writing style based on ratings and product features.',
    'abstract_generation': 'Academic abstract generation: Create comprehensive abstracts for research papers based on titles and key research items, incorporating domain-specific knowledge and writing style.',
    'PRISM': 'Conversational AI preference alignment: Understand user preferences for AI assistant responses based on conversation history and feedback patterns.',
    'prism_data': 'Conversational AI preference alignment: Understand user preferences for AI assistant responses based on conversation history and feedback patterns.',
}

def get_task_specific_prompt_template(task_name, prompts_dir="./data_p13n/prompts", use_unified_template=False):
    """
    Get the appropriate prompt template based on task name.
    Maps task names to specific prompt template files.
    If use_unified_template is True, uses unified template with task description.
    """
    task_prompt_mapping = {
        # LaMP tasks
        'citation': 'citation_profile_prompt.md',
        'movie': 'movie_profile_prompt.md', 
        'news_cat': 'news_cat_profile_prompt.md',
        'news_headline': 'news_headline_profile_prompt.md',
        'product': 'product_profile_prompt.md',
        'scholarly_title': 'scholarly_profile_prompt.md',
        'tweet': 'tweet_profile_prompt.md',
        
        # LongLaMP tasks  
        'topic_writing': 'topic_writing_profile_prompt.md',
        'product_review': 'product_review_profile_prompt.md',
        'abstract_generation': 'abstract_generation_profile_prompt.md',
        
        # PRISM tasks
        'PRISM': 'prism_profile_prompt.md',
        'prism_data': 'prism_profile_prompt.md',
    }
    
    # Extract the base task name from complex task names
    base_task_name = task_name
    
    # Handle LaMP processed data naming patterns
    if 'LaMP_processed_' in task_name:
        # Extract task type from LaMP_processed_taskname_data format
        parts = task_name.replace('LaMP_processed_', '').split('_')
        if len(parts) >= 1:
            base_task_name = parts[0]
            # Handle news_cat and news_headline cases
            if base_task_name == 'news':
                if len(parts) >= 2 and parts[1] == 'cat':
                    base_task_name = 'news_cat'
                elif len(parts) >= 2 and parts[1] == 'headline':
                    base_task_name = 'news_headline'
            # Handle scholarly_title case
            elif len(parts) >= 2 and parts[0] == 'scholarly' and parts[1] == 'title':
                base_task_name = 'scholarly_title'
    
    # Handle LongLaMP naming patterns  
    elif 'LongLaMP_' in task_name:
        # Extract task type from LongLaMP_taskname_data format
        base_task_name = task_name.replace('LongLaMP_', '').replace('_data', '')
    
    # If using unified template, load it and inject task description
    if use_unified_template:
        unified_template_path = os.path.join(prompts_dir, 'unified_profile_prompt.md')
        
        # Ensure unified template exists
        if not os.path.exists(unified_template_path):
            print(f"Error: Unified template not found at {unified_template_path}")
            raise FileNotFoundError(f"Unified template not found: {unified_template_path}")
        
        # Load the unified template
        with open(unified_template_path, 'r') as f:
            template_content = f.read()
        
        # Get task description from the mapping
        task_description = task2discription.get(base_task_name, 'general user profiling')
        
        # Create a custom template class that will render both placeholders at once
        from jinja2 import Template
        
        class UnifiedTemplate:
            def __init__(self, template_content, task_description):
                self.template = Template(template_content)
                self.task_description = task_description
            
            def render(self, user_history):
                return self.template.render(
                    task_description=self.task_description,
                    user_history=user_history
                )
        
        return UnifiedTemplate(template_content, task_description)
    
    # Original logic for task-specific templates
    # Get the appropriate template file
    template_file = task_prompt_mapping.get(base_task_name, 'default_profile_prompt.md')
    template_path = os.path.join(prompts_dir, template_file)
    
    # Fallback to default if specific template doesn't exist
    if not os.path.exists(template_path):
        print(f"Warning: Task-specific template for '{base_task_name}' not found at {template_path}. Using default template.")
        template_path = os.path.join(prompts_dir, 'default_profile_prompt.md')
    
    return load_prompt_template(template_path)


def detect_history_format(sample_history_items, limit=10):
    """Detect the format of history items by examining a sample"""
    if not sample_history_items:
        return []
    
    # Look at up to 'limit' items to detect the format
    sample_items = sample_history_items[:limit]
    
    # Count frequency of keys across all sample items
    key_frequencies = {}
    for item in sample_items:
        if isinstance(item, dict):
            for key in item.keys():
                key_frequencies[key] = key_frequencies.get(key, 0) + 1
    
    # Sort keys by frequency (most common first)
    common_keys = sorted(key_frequencies.items(), key=lambda x: x[1], reverse=True)
    
    # print(f"Detected history keys (by frequency): {common_keys}")
    return [key for key, freq in common_keys]


def history_to_string(history_items, file_name="", detected_keys=None):
    """Convert history items to a formatted string for the prompt, handling different formats"""
    if not history_items:
        return "No history available."
    
    # Detect the format of history items if not provided
    if detected_keys is None:
        detected_keys = detect_history_format(history_items)
    
    result = []
    for item in history_items:
        if not isinstance(item, dict):
            continue
            
        # Use the new formatting function that includes all keys
        formatted_item = format_history_item_truncated(item, detected_keys)
        if formatted_item:
            result.append(formatted_item)
    
    if not result:
        return "No valid history available."
    
    # Add file context in the header
    file_context = f" (from {file_name})" if file_name else ""
    header = f"\n{'='*50}\nUser History{file_context}\n{'='*50}\n"
    separator = "\n" + "-" * 50 + "\n"
    
    return header + separator.join(result) + "\n" + "=" * 50


def create_profile_requests(
    data, prompt_template, model_name, file_name="", task_name=None, max_context_length=32768,
    unified_single_profile_mode=False
):
    """
    Create LLM requests for profile generation with expanding history ranges.
    If unified_single_profile_mode is True, only generate profile for the entire history ([0, -1]).
    """
    requests = []
    print(f"Creating profile requests for {len(data)} users...")
    for user_data in tqdm(data, desc="Preparing LLM inputs"):
        user_id = user_data.get('user_id', 'unknown')
        history = user_data.get('history', [])
        if not history:
            continue

        if unified_single_profile_mode:
            # Only generate profile for the entire history
            i = len(history) - 1
            visible_history = history[0:i+1]
            visible_history = truncate_history_to_fit_context(
                visible_history, prompt_template, max_context_length, file_name
            )
            if not visible_history:
                print(f"Skipping profile generation for {user_id}_{i} - no history fits context limit")
                continue
            user_history_text = history_to_string(visible_history, file_name)
            prompt = prompt_template.render(user_history=user_history_text)
            # print(prompt)
            # input()
            profile_id = f"{user_id}_{i}"
            messages = [{"role": "user", "content": prompt}]
            request = {
                "custom_id": profile_id,
                "body": {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.6,
                    "max_tokens": 2000,
                    "top_p": 0.9,
                }
            }
            if task_name is not None:
                request["task_name"] = task_name
            requests.append(request)
        else:
            # Generate profiles for expanding history ranges [0:4] to [0:-2]
            # Start from index 4 and go to the second-to-last item
            start_idx = 4
            end_idx = len(history) - 1  # -2 in Python slicing means exclude last item, so -1 for second-to-last
            
            if start_idx >= len(history):
                # If history is too short, just use what we have
                start_idx = min(3, len(history) - 1)
            
            for i in range(start_idx, end_idx + 1):
                if i < len(history):
                    # Create history slice [0:i+1] to include index i
                    visible_history = history[0:i+1]
                    
                    # Truncate history if it exceeds context length
                    visible_history = truncate_history_to_fit_context(
                        visible_history, prompt_template, max_context_length, file_name
                    )
                    
                    # Skip if no history remains after truncation
                    if not visible_history:
                        print(f"Skipping profile generation for {user_id}_{i} - no history fits context limit")
                        continue
                    
                    # Generate the prompt with file context
                    user_history_text = history_to_string(visible_history, file_name)
                    prompt = prompt_template.render(user_history=user_history_text)
                    
                    # Create a unique ID for this profile generation task
                    profile_id = f"{user_id}_{i}"
                    
                    messages = [{"role": "user", "content": prompt}]
                    
                    request = {
                        "custom_id": profile_id,
                        "body": {
                            "model": model_name,
                            "messages": messages,
                            "temperature": 0.6,
                            "max_tokens": 2000,
                            "top_p": 0.9,
                        }
                    }
                    # Add task_name for output file routing
                    if task_name is not None:
                        request["task_name"] = task_name
                    requests.append(request)
    return requests


def create_profile_requests_prism(data, prompt_template, model_name, file_name="", task_name=None, max_context_length=32768):
    """Create LLM requests for profile generation from PRISM data with expanding conversation ranges"""
    requests = []
    
    print(f"Creating PRISM profile requests for {len(data)} users...")
    for user_data in tqdm(data, desc="Preparing PRISM LLM inputs"):
        user_id = user_data.get('user_id', 'unknown')
        conversations = user_data.get('conversations', [])
        
        if not conversations:
            continue
        
        # For PRISM, each conversation is treated as a history item
        # Generate profiles for expanding conversation ranges [0:4] to [0:-2]
        start_idx = 4
        end_idx = len(conversations) - 1  # -2 in Python slicing means exclude last item, so -1 for second-to-last
        
        if start_idx >= len(conversations):
            # If conversations list is too short, just use what we have
            start_idx = min(3, len(conversations) - 1)
        
        for i in range(start_idx, end_idx + 1):
            if i < len(conversations):
                # Create conversations slice [0:i+1] to include index i
                visible_conversations = conversations[0:i+1]
                
                # Convert conversations to history format
                history_items = []
                for conv in visible_conversations:
                    # Each conversation becomes a history item
                    conv_item = {
                        'conversation_id': conv.get('conversation_id', ''),
                        'conversation_type': conv.get('conversation_type', ''),
                        'opening_prompt': conv.get('opening_prompt', ''),
                        'conversation_turns': conv.get('conversation_turns', 0),
                        'conversation_summary': format_conversation_history(conv.get('conversation_history', []))
                    }
                    history_items.append(conv_item)
                
                # Truncate history if it exceeds context length
                history_items = truncate_history_to_fit_context(
                    history_items, prompt_template, max_context_length, file_name
                )
                
                # Skip if no history remains after truncation
                if not history_items:
                    print(f"Skipping profile generation for {user_id}_{i} - no history fits context limit")
                    continue
                
                # Generate the prompt with file context
                user_history_text = history_to_string(history_items, file_name)
                prompt = prompt_template.render(user_history=user_history_text)
                
                # Create a unique ID for this profile generation task
                profile_id = f"{user_id}_{i}"
                
                messages = [{"role": "user", "content": prompt}]
                
                request = {
                    "custom_id": profile_id,
                    "body": {
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.6,
                        "max_tokens": 2000,
                        "top_p": 0.9,
                    }
                }
                # Add task_name for output file routing
                if task_name is not None:
                    request["task_name"] = task_name
                requests.append(request)
    
    return requests


def format_conversation_history(conversation_history):
    """Format a conversation history into a readable summary"""
    if not conversation_history:
        return "No conversation content"
    
    formatted_turns = []
    for turn in conversation_history:
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        turn_num = turn.get('turn', '')
        
        if role == 'user':
            formatted_turns.append(f"User (Turn {turn_num}): {content}")
        elif role == 'model':
            # Only include chosen model responses to keep it concise
            if turn.get('if_chosen', False):
                formatted_turns.append(f"Assistant (Turn {turn_num}): {content}")
    
    return "\n".join(formatted_turns)


async def generate_profiles_for_file(input_file, output_file, model_name, prompt_template, output_path_dict=None, task_name=None, max_context_length=32768, batch_size=200, unified_single_profile_mode=False, prompts_dir="./data_p13n/prompts", use_unified_template=False):
    """Generate user profiles for a single data file"""
    print(f"Processing {input_file}...")
    
    # Load data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} users from {input_file}")
    
    # Get file name for context
    file_name = Path(input_file).name
    
    # Use task-specific prompt template if task_name is provided
    if task_name:
        try:
            prompt_template = get_task_specific_prompt_template(task_name, prompts_dir, use_unified_template)
            template_type = "unified" if use_unified_template else "task-specific"
            print(f"Using {template_type} prompt template for task: {task_name}")
        except Exception as e:
            print(f"Warning: Could not load template for {task_name}: {e}. Using default template.")
    else:
        print("Using provided default prompt template")
    
    # Check for existing output and completed profiles
    completed = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed[record['custom_id']] = record
        print(f"Found {len(completed)} completed profiles, will skip them")
    
    # Always use unified_single_profile_mode for all tasks except PRISM
    unified_single_profile_mode_flag = True

    # Create requests
    requests = create_profile_requests(
        data, prompt_template, model_name, file_name, task_name=task_name,
        max_context_length=max_context_length,
       unified_single_profile_mode=unified_single_profile_mode_flag
    )
    
    # Filter out already completed requests
    pending_requests = [req for req in requests if req['custom_id'] not in completed]
    
    print(f"Created {len(requests)} total requests, {len(pending_requests)} pending")
    
    if not pending_requests:
        print("All profiles already generated!")
        return
    
    # Set up OpenAI client
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )
    
    # Generate profiles
    await call_llm_in_parallel(
        client, 
        pending_requests, 
        model_name, 
        output_path=output_path_dict,  # output_path_dict: {task_name: output_file}
        batch_size=batch_size
    )
    
    print(f"Profile generation completed for {input_file}")


async def generate_profiles_for_prism_file(input_file, output_file, model_name, prompt_template, output_path_dict=None, task_name=None, max_context_length=32768, batch_size=200, prompts_dir="./data_p13n/prompts", use_unified_template=False):
    """Generate user profiles for PRISM data file"""
    print(f"Processing PRISM data from {input_file}...")
    
    # Load data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} users from {input_file}")
    
    # Get file name for context
    file_name = Path(input_file).name
    
    # Use task-specific prompt template for PRISM
    try:
        prompt_template = get_task_specific_prompt_template("PRISM", prompts_dir, use_unified_template)
        template_type = "unified" if use_unified_template else "task-specific"
        print(f"Using {template_type} prompt template for PRISM")
    except Exception as e:
        print(f"Warning: Could not load PRISM template: {e}. Using default template.")
    
    # Check for existing output and completed profiles
    completed = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed[record['custom_id']] = record
        print(f"Found {len(completed)} completed profiles, will skip them")
    
    # Create requests using PRISM-specific function
    requests = create_profile_requests_prism(data, prompt_template, model_name, file_name, task_name=task_name, max_context_length=max_context_length)
    
    # Filter out already completed requests
    pending_requests = [req for req in requests if req['custom_id'] not in completed]
    
    print(f"Created {len(requests)} total requests, {len(pending_requests)} pending")
    
    if not pending_requests:
        print("All profiles already generated!")
        return
    
    # Set up OpenAI client
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )
    
    # Generate profiles
    await call_llm_in_parallel(
        client, 
        pending_requests, 
        model_name, 
        output_path=output_path_dict,  # output_path_dict: {task_name: output_file}
        batch_size=batch_size
    )
    
    print(f"Profile generation completed for {input_file}")


def organize_profiles_by_user_and_split(output_files_dict):
    """Organize generated profiles by user and split type (random_train/random_test)"""
    result = {}
    
    for split_type, output_file in output_files_dict.items():
        if not os.path.exists(output_file):
            result[split_type] = {}
            continue
            
        profiles_by_user = {}
        user_max_idx = {}  # Track the highest end_idx for each user
        
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    custom_id = record['custom_id']
                    content = record['content']
                    
                    # Parse the custom_id to extract user_id and end_idx
                    # Format: {user_id}_{end_idx}
                    parts = custom_id.rsplit('_', 1)
                    if len(parts) == 2:
                        user_id = parts[0]
                        end_idx = int(parts[1])
                        
                        # Keep the profile with the highest end_idx for each user
                        if user_id not in profiles_by_user or end_idx > user_max_idx.get(user_id, -1):
                            profiles_by_user[user_id] = content
                            user_max_idx[user_id] = end_idx
        
        result[split_type] = profiles_by_user
    
    return result


async def main():
    parser = argparse.ArgumentParser(description="Generate user profiles for LaMP, LongLaMP, and PRISM data")
    parser.add_argument("--lamp_dir", type=str, default="./data_p13n/LaMP",
                       help="Directory containing LaMP data files")
    parser.add_argument("--longlamp_dir", type=str, default="./data_p13n/LongLaMP",
                       help="Directory containing LongLaMP data files")
    parser.add_argument("--prism_dir", type=str, default="./data_p13n/PRISM",
                       help="Directory containing PRISM data files")
    parser.add_argument("--output_dir", type=str, default="text-to-lora/data_p13n/user_profiles",
                       help="Directory to save generated profiles")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                       help="Model name for profile generation")
    parser.add_argument("--prompts_dir", type=str, default="./data_p13n/prompts",
                       help="Directory containing task-specific prompt template files")
    parser.add_argument("--max_context_length", type=int, default=32768,
                       help="Maximum context length for the model (default: 32768)")
    parser.add_argument("--batch_size", type=int, default=200,
                       help="Batch size for profile generation (default: 200)")
    parser.add_argument("--use_unified_template", action="store_true",
                       help="Use unified prompt template with task descriptions instead of task-specific templates")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create prompts directory if it doesn't exist and ensure default template exists
    prompts_dir = Path(args.prompts_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Create default template if it doesn't exist
    default_template_path = prompts_dir / "default_profile_prompt.md"
    if not default_template_path.exists():
        print(f"Creating default prompt template at {default_template_path}")
        with open(default_template_path, 'w') as f:
            f.write("""# Instruction

Generate a comprehensive and detailed user profile based solely on the provided user history data. Ensure coverage of the following key aspects, including but not limited to:

1. **User Preferences:**
   - Explicitly stated interests and feedback
   - Implicit preferences inferred from user interactions, choices, and behaviors
   - Preferences across various content types, products, or activities

2. **Behavioral Patterns:**
   - Regular behaviors and interaction habits
   - Frequency, duration, and timing of user activities
   - Identifiable trends, patterns, or deviations over time
   - Engagement levels and responsiveness to specific types of interactions

3. **Demographic Information:**
   - Inferred or explicitly available demographic details such as age, gender, education level, occupation, geographic location, income range, and other pertinent personal characteristics

Aim for accuracy and comprehensiveness, deriving insights strictly from the provided historical data.

# User History Data

{{ user_history }}

# Output Format

Output the user profile strictly in plain text. Do not include explanations, introductions, headings, bullet points, or any formatting structure.""")
    
    # Use a placeholder template for function calls (will be replaced by task-specific templates)
    prompt_template = None
    
    # Process both directories
    directories = {
        "LaMP": args.lamp_dir,
        "LongLaMP": args.longlamp_dir,
        "PRISM": args.prism_dir
    }
    
    for dataset_name, dataset_dir in directories.items():
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            print(f"Dataset directory {dataset_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {dataset_name} dataset from {dataset_path}...")
        
        if dataset_name == "PRISM":
            # Handle PRISM dataset differently
            prism_data_file = dataset_path / "PRISM_data.jsonl"
            if not prism_data_file.exists():
                print(f"PRISM data file {prism_data_file} does not exist, skipping...")
                continue
            
            if prism_data_file.stat().st_size == 0:  # Skip empty files
                print(f"Skipping empty file: {prism_data_file}")
                continue
            
            print(f"Found PRISM data file: {prism_data_file.name}")
            
            # Prepare output files for PRISM
            task_name = "PRISM"
            temp_output_file = output_dir / f"{task_name}_temp_profiles.jsonl"
            final_output_file = output_dir / f"{task_name}_profiles.json"
            
            # Check if final structured output already exists
            if final_output_file.exists():
                print(f"Final structured output {final_output_file} already exists. Skipping PRISM dataset.")
                continue
            
            # Prepare output path dict for PRISM
            output_path_dict = {task_name: str(temp_output_file)}
            
            # Generate profiles for PRISM
            try:
                await generate_profiles_for_prism_file(
                    str(prism_data_file),
                    str(temp_output_file),
                    args.model_name,
                    prompt_template,
                    output_path_dict=output_path_dict,
                    task_name=task_name,
                    max_context_length=args.max_context_length,
                    batch_size=args.batch_size,
                    prompts_dir=args.prompts_dir,
                    use_unified_template=args.use_unified_template
                )
                
                # Organize profiles by user (PRISM only has one "split")
                structured_profiles = {"prism_data": {}}
                
                if temp_output_file.exists():
                    profiles_by_user = {}
                    user_max_idx = {}  # Track the highest end_idx for each user
                    
                    with open(temp_output_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                custom_id = record['custom_id']
                                content = record['content']
                                
                                # Parse the custom_id to extract user_id and end_idx
                                # Format: {user_id}_{end_idx}
                                parts = custom_id.rsplit('_', 1)
                                if len(parts) == 2:
                                    user_id = parts[0]
                                    end_idx = int(parts[1])
                                    
                                    # Keep the profile with the highest end_idx for each user
                                    if user_id not in profiles_by_user or end_idx > user_max_idx.get(user_id, -1):
                                        profiles_by_user[user_id] = content
                                        user_max_idx[user_id] = end_idx
                    
                    structured_profiles["prism_data"] = profiles_by_user
                
                # Save the structured output for PRISM
                with open(final_output_file, 'w') as f:
                    json.dump(structured_profiles, f, indent=2)
                
                print(f"Saved PRISM profiles to {final_output_file}")
                print(f"Total PRISM users processed: {len(structured_profiles['prism_data'])}")
                
            except Exception as e:
                print(f"Error processing PRISM data: {e}")
                continue
        else:
            # Handle LaMP and LongLaMP datasets (existing logic)
            # Find all .jsonl files containing "_data.jsonl" in filename
            all_jsonl_files = list(dataset_path.glob("*.jsonl"))
            filtered_files = [f for f in all_jsonl_files 
                            if "_data.jsonl" in f.name]
            
            if not filtered_files:
                print(f"No _data.jsonl files found in {dataset_path}")
                continue
                
            print(f"Found {len(filtered_files)} relevant data JSONL files: {[f.name for f in filtered_files]}")
            
            # Group files by task (base name without _data suffix)
            tasks = {}
            for data_file in filtered_files:
                if data_file.stat().st_size == 0:  # Skip empty files
                    print(f"Skipping empty file: {data_file}")
                    continue
                    
                # Extract base task name by removing _data suffix
                file_name = data_file.stem
                if "_data" in file_name:
                    # Find the last occurrence of "_data" and remove it along with any suffix
                    data_index = file_name.rfind("_data")
                    task_name = file_name[:data_index]
                    split_type = "data"
                else:
                    continue
                    
                if task_name not in tasks:
                    tasks[task_name] = {}
                tasks[task_name][split_type] = data_file
            
            print(f"Grouped into {len(tasks)} tasks: {list(tasks.keys())}")
            
            # Process each task
            for task_name, task_files in tqdm(tasks.items(), desc=f"Processing {dataset_name} tasks"):
                print(f"\nProcessing task: {task_name}")
                
                # Prepare output files for this task
                output_path_dict = {}
                temp_output_files = {}
                
                for split_type, data_file in task_files.items():
                    temp_output_file = output_dir / f"{task_name}_{split_type}_temp_profiles.jsonl"
                    output_path_dict[data_file.stem] = str(temp_output_file)
                    temp_output_files[split_type] = str(temp_output_file)
                
                # Check if final structured output for this task already exists
                final_output_file = output_dir / f"{task_name}_profiles.json"
                if final_output_file.exists():
                    print(f"Final structured output {final_output_file} already exists. Skipping task {task_name}.")
                    continue

                # For data files, always use unified_single_profile_mode
                split_unified_flags = {}
                for split_type in task_files:
                    split_unified_flags[split_type] = True

                # Generate profiles for each file in this task
                for split_type, data_file in tqdm(task_files.items(), desc=f"Processing {task_name} files", leave=False):
                    try:
                        await generate_profiles_for_file(
                            str(data_file), 
                            output_path_dict[data_file.stem], 
                            args.model_name, 
                            prompt_template,
                            output_path_dict=output_path_dict,
                            task_name=data_file.stem,
                            max_context_length=args.max_context_length,
                            batch_size=args.batch_size,
                            unified_single_profile_mode=split_unified_flags[split_type],
                            prompts_dir=args.prompts_dir,
                            use_unified_template=args.use_unified_template
                        )
                    except Exception as e:
                        print(f"Error processing {data_file}: {e}")
                        continue
                
                # Organize profiles by user and split type
                structured_profiles = organize_profiles_by_user_and_split(temp_output_files)
                
                # Save the combined structured output for this task
                with open(final_output_file, 'w') as f:
                    json.dump(structured_profiles, f, indent=2)
                
                print(f"Saved combined profiles to {final_output_file}")
                
                # Count total users
                total_users = 0
                for split_type, user_profiles in structured_profiles.items():
                    total_users += len(user_profiles)
                    print(f"  {split_type}: {len(user_profiles)} users")
                
                print(f"Total data users processed for {task_name}: {total_users}")
                
                # Do NOT delete temp checkpoint files; they are used for checkpoint resuming.
                # for temp_file in temp_output_files.values():
                #     if os.path.exists(temp_file):
                #         os.remove(temp_file)
    
    print("\nAll profile generation completed!")


if __name__ == "__main__":
    asyncio.run(main()) 