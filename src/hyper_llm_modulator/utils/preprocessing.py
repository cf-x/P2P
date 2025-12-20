"""
Preprocessing functions for various datasets.

This module includes preprocessing functions for:

1. LOL (Lots of LoRAs) datasets: Parse task definitions and problems
2. ARC datasets: Format multiple choice questions  
3. MBPP datasets: Format code problems with test assertions
4. PWC datasets: Parse context-query-answer format
5. BookSum datasets: Add random summary questions

PERSONALIZATION DATASETS (LaMP & LongLaMP):
6. LaMP Movie Tagging: Parse movie descriptions and tag prediction instructions
7. LaMP Citation: Parse paper titles and reference options
8. LaMP Product Rating: Parse product reviews and rating instructions  
9. LaMP Tweet: Parse original tweets and paraphrasing instructions
10. LaMP News Headline/Category: Parse news descriptions and categorization instructions
11. LaMP Scholarly Title: Parse paper information for title generation
12. LongLaMP Product Review: Parse product info, ratings, and review summaries
13. LongLaMP Abstract Generation: Parse paper titles and key items for abstract generation  
14. LongLaMP Topic Writing: Parse topic prompts for content generation

The preprocessing functions split the "input" field into fine-grained components:
- instruction: The task instruction/prompt
- description/context: The main content to process
- Additional task-specific fields (paper_title, rating, references, etc.)
- query/answer: Standardized fields for compatibility

Usage:
    preprocessing_fn = get_preprocessing_fn("lamp_movie")
    processed_example = preprocessing_fn(raw_example)
"""

import random
from typing import Callable, Literal
import re
from pathlib import Path

import pandas as pd

def is_gen_profile_only_task(task_name: str | None) -> bool:
    """Return True when only generated profile text should be embedded."""
    if not task_name:
        return False

    name = str(task_name).lower()
    tokens = [token for token in re.split(r"[\/_-]", name) if token]

    if any(token in {"personalreddit", "prism", "aloe"} for token in tokens):
        return True

    if any(token == "ec" for token in tokens):
        return True

    return False


SUM_Q = [
    "Summarize the text.",
    "Please summarize the text.",
    "Can you summarize the text?",
    "Give a summary of the text.",
    "Summarize the text for me.",
    "Please give a summary of the text.",
    "Can you give a summary of the text?",
    "Summarize.",
    "Summarize the text, please.",
    "Summarize the text, thank you.",
    "Give me a summary of the text.",
    "Please give me a summary of the text.",
]


REPO_ROOT = Path(__file__).resolve().parents[3]
CHAT_TEMPLATE_BASE_DIR = REPO_ROOT / "chat_templates"


def get_preprocessing_fn(ds_name):
    f = lambda x: x
    if ds_name.startswith("lol_"):

        def f(example):
            txt = example["input"]
            task_def = txt.split("Definition: ")[1].split("\n\nPositive Example")[0]
            task_def += " Please complete the task without any explanation."
            if len(example["output"]) > 1:
                task_def += "\nThe answer should be a comma-separated list of possible completions."
            problem = txt.split("Now complete the following example -")[1].split("Input: ")[1].split("\nOutput:")[0]
            answer = ", ".join(example["output"])
            return dict(task_def=task_def, problem=problem, answer=answer)

    if ds_name.startswith("arc_"):
        ABCD = ["A", "B", "C", "D"]

        def f(example):
            choices = example["choices"]
            assert len(choices["text"]) == len(choices["label"])
            n_to_fill = 4 - len(choices["text"])
            if len(choices["text"]) < 4:
                choices["text"] += ["N/A"] * n_to_fill
            if len(choices["label"]) < 4:
                if choices["label"][0].isdigit():
                    choices["label"] += [str(len(choices["label"]) + i + 1) for i in range(n_to_fill)]
                else:
                    choices["label"] += [ABCD[len(choices["label"]) + i] for i in range(n_to_fill)]
            example["choices"] = choices
            return example

    if ds_name.startswith("mbpp"):
        # for training an oracle lora on mbpp
        def f(example):
            example["assertions"] = "\n".join(example["test_list"])
            return example

    if "pwc" in ds_name:

        def f(example):
            return dict(context=example["input"], query=example["prompt"], answer=example["answer"])

    if "booksum" in ds_name:

        def f(example):
            context = example["chapter"].strip()
            query = random.sample(SUM_Q, 1)[0]
            return dict(context=context, query=query, answer=example["summary_text"])

    # LaMP Movie Tagging: Parse movie description and instruction
    if ds_name == "lamp_movie" or "movie" in ds_name:
        def f(example):
            input_text = example["input"]
            # Split on "description:" to separate instruction and movie description
            if "description:" in input_text:
                parts = input_text.split("description:", 1)
                instruction = parts[0].strip()
                description = parts[1].strip()
            else:
                # Fallback: treat entire input as instruction
                instruction = input_text
                description = ""
            
            return dict(
                instruction=instruction,
                description=description,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=description,  # alias for compatibility
                # query=instruction,    # alias for compatibility
                answer=example["output"]
            )

    # LaMP Citation: Parse paper title and reference options
    if ds_name == "lamp_citation" or "citation" in ds_name:
        def f(example):
            input_text = example["input"]
            # Extract paper title and options
            # Pattern: 'For an author who has written the paper with the title "TITLE", which reference is related?'
            title_match = re.search(r'title "([^"]+)"', input_text)
            paper_title = title_match.group(1) if title_match else ""
            
            # Extract options [1] and [2]
            ref_pattern = r'\[(\d+)\]: "([^"]+)"'
            options = re.findall(ref_pattern, input_text)
            option_dict = {option[0]: option[1] for option in options}
            
            # INSERT_YOUR_CODE
            # Format the options as a string: [1]: "..." [2]: "..."
            options_str = "\n".join([f'[{k}]: "{v}"' for k, v in option_dict.items()])
            # Extract base instruction
            instruction = re.split(r'\[1\]:', input_text)[0].strip()
            
            return dict(
                instruction=instruction,
                paper_title=paper_title,
                options_str=options_str,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=paper_title,  # alias for compatibility
                # query=instruction,    # alias for compatibility
                answer=example["output"]
            )

    # LaMP Product Rating: Parse review text and instruction
    if ds_name == "lamp_product" or ("product" in ds_name and "lamp" in ds_name.lower()):
        def f(example):
            input_text = example["input"]
            # Split on "review:" to separate instruction and review text
            if "review:" in input_text:
                parts = input_text.split("review:", 1)
                instruction = parts[0].strip()
                review_text = parts[1].strip()
            else:
                # Fallback: treat entire input as instruction
                instruction = input_text
                review_text = ""
            
            return dict(
                instruction=instruction,
                review_text=review_text,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=review_text,  # alias for compatibility
                # query=instruction,    # alias for compatibility
                answer=example["output"]
            )

    # LaMP Tweet Paraphrasing: Parse original tweet and instruction
    if ds_name == "lamp_tweet" or "tweet" in ds_name:
        def f(example):
            input_text = example["input"]
            # Extract the tweet after the colon
            if "before or after it:" in input_text:
                parts = input_text.split("before or after it:", 1)
                instruction = parts[0].strip()
                original_tweet = parts[1].strip()
            else:
                instruction = input_text
                original_tweet = ""
            
            return dict(
                instruction=instruction,
                original_tweet=original_tweet,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=original_tweet,  # alias for compatibility
                # query=instruction,       # alias for compatibility
                answer=example["output"]
            )

    # LaMP News Headline: Parse description and instruction
    if ds_name == "lamp_news_headline" or "news_headline" in ds_name:
        def f(example):
            input_text = example["input"]
            # Split on "description:" to separate instruction and news description
            if "following article:" in input_text:
                parts = input_text.split("following article:", 1)
                instruction = parts[0].strip()
                article = parts[1].strip()
            else:
                instruction = input_text
                article = ""
            
            return dict(
                instruction=instruction,
                article=article,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=description,  # alias for compatibility
                # query=instruction,    # alias for compatibility
                answer=example["output"]
            )

    # LaMP News Category: Parse description and instruction  
    if ds_name == "lamp_news_cat" or "news_cat" in ds_name:
        def f(example):
            input_text = example["input"]
            # Split on "description:" to separate instruction and news description
            if "article:" in input_text:
                parts = input_text.split("article:", 1)
                instruction = parts[0].strip()
                article = parts[1].strip()
            else:
                instruction = input_text
                article = ""
            
            return dict(
                instruction=instruction,
                article=article,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=description,  # alias for compatibility
                # query=instruction,    # alias for compatibility
                answer=example["output"]
            )

    # LaMP Scholarly Title: Parse paper info and instruction
    if ds_name == "lamp_scholarly_title" or "scholarly_title" in ds_name:
        def f(example):
            input_text = example["input"]

            instruction = "Generate a title for the following abstract of a paper: "
            abstract = input_text.replace(instruction, "").strip()
            # This would need to be customized based on the actual data structure
            # For now, treating as generic instruction-context split
            return dict(
                instruction=instruction,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context="",  # placeholder
                abstract=abstract,
                answer=example["output"]
            )

    # LongLaMP Product Review Generation: Parse product info, rating, and summary
    if ds_name == "longlamp_product_review" or ("product_review" in ds_name and "long" in ds_name.lower()):
        def f(example):
            input_text = example["input"]
            
            # Extract rating
            rating_match = re.search(r'rating of "([^"]+)"', input_text)
            rating = rating_match.group(1) if rating_match else ""
            
            # Extract product description
            product_match = re.search(r'product with description "([^"]+)"', input_text)
            product_description = product_match.group(1) if product_match else ""
            
            # Extract review summary
            summary_match = re.search(r'summary of the review text is "([^"]+)"', input_text)
            review_summary = summary_match.group(1) if summary_match else ""
            
            # Extract base instruction
            instruction = input_text.split('Generate the review text')[0] + 'Generate the review text'
            
            return dict(
                instruction=instruction,
                product_description=product_description,
                rating=rating,
                review_summary=review_summary,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=f"Product: {product_description}, Rating: {rating}, Summary: {review_summary}",
                # query=instruction,
                answer=example["output"]
            )

    # LongLaMP Abstract Generation: Parse title and key items
    if ds_name == "longlamp_abstract_generation" or ("abstract_generation" in ds_name and "long" in ds_name.lower()):
        def f(example):
            input_text = example["input"]
            
            # Extract paper title
            title_match = re.search(r'title "([^"]+)"', input_text)
            paper_title = title_match.group(1) if title_match else ""
            
            # Extract numbered items
            items_section = input_text.split("using the following items:", 1)
            if len(items_section) > 1:
                items_text = items_section[1].strip()
                # Extract numbered items
                item_pattern = r'\d+\.\s*([^\n]+)'
                items = re.findall(item_pattern, items_text)
            else:
                items = []
            
            instruction = f"Generate an abstract for the title \"{paper_title}\""
            
            return dict(
                instruction=instruction,
                paper_title=paper_title,
                key_items=items,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=f"Title: {paper_title}, Key items: {', '.join(items)}",
                # query=instruction,
                answer=example["output"]
            )

    # LongLaMP Topic Writing: Parse topic and writing constraints
    if ds_name == "longlamp_topic_writing" or ("topic_writing" in ds_name and "long" in ds_name.lower()):
        def f(example):
            input_text = example["input"]
            
            # Extract the topic/prompt after "reddit post"
            if "reddit post" in input_text:
                parts = input_text.split(" for a reddit post", 1)
                instruction = parts[0].strip() + " for a reddit post"
                topic_prompt = parts[1].strip()
            else:
                raise ValueError(f"Unexpected input format: {input_text}")
                
                # Fallback to splitting on common patterns
                # instruction = "Generate content"
                # topic_prompt = input_text
            
            return dict(
                instruction=instruction,
                topic_prompt=topic_prompt,
                profile_text=example["profile_text"],
                profile_retrieval_k1=example["profile_retrieval_k1"],
                profile_retrieval_k2=example["profile_retrieval_k2"],
                profile_retrieval_k4=example["profile_retrieval_k4"],
                profile_all_history=example["profile_all_history"],
                # context=topic_prompt,  # topic is the context
                # query=instruction,
                answer=example["output"]
            )

    # OpinionQA: Parse opinion questions with demographic profiles
    if ds_name == "opinionqa" or "opinionqa" in ds_name.lower():
        def f(example):
            input_text = example["input"]
            
            # Split the input to extract question and options
            if "Question:" in input_text and "Options:" in input_text:
                parts = input_text.split("Options:", 1)
                question = parts[0].replace("Question:", "").strip()
                options_part = parts[1].split("Answer:")[0].strip()
                
                instruction = "Answer the following multiple choice question based on your personal opinions and demographic background."
                # context = f"Question: {question}\nOptions: {options_part}"
            else:
                # Fallback
                instruction = "Answer the following question."
                # context = input_text
            
            return dict(
                instruction=instruction,
                # context=context,
                user_id=example["user_id"],
                question_id=example["question_id"],
                question=question,
                options=options_part,
                profile_text=example["profile_text"],
                answer=example["output"]
            )
            
    print(ds_name)
    # EC (Empathetic Conversations): Parse news commentary with personality profiles
    if ds_name == "ec" or "ec" in ds_name.lower():
        def f(example):
            input_text = example["input"]
            
            instruction = "Read the following news article and provide your personal commentary or reaction."
            # context = input_text.strip()
            
            return dict(
                instruction=instruction,
                # context=context,
                article=input_text,
                user_id=example["user_id"],
                question_id=example["question_id"],
                profile_text=example["profile_text"],
                answer=example["output"]
            )

    # PersonalReddit: Parse personal conversation responses
    if ds_name == "personalreddit" or "personalreddit" in ds_name.lower():
        def f(example):
            input_text = example["input"]
            
            instruction = "Respond to the following conversation or question in a personal and authentic way that reflects your background and personality."
            # context = input_text.strip()
            
            return dict(
                instruction=instruction,
                # context=context,
                input_text=input_text,
                user_id=example["user_id"],
                question_id=example["question_id"],
                profile_text=example["profile_text"],
                answer=example["output"]
            )

    # PRISM and ALOE: Parse chat conversations for apply_chat_template processing
    if "prism" in ds_name.lower() or "aloe" in ds_name.lower():
        def f(example):
            input_conversation = example["input"]
            
            return dict(
                conversation=input_conversation,  # Store the raw conversation list for apply_chat_template
                user_id=example["user_id"],
                question_id=example["question_id"],
                profile_text=example["profile_text"],
                answer=example["output"],
                # Special flag to indicate this needs chat template processing
                is_prism_conversation=True  # Keep same flag name for compatibility
            )

    return f


def add_full_stop(s):
    s = s.strip()
    # check if s ends with . or .*
    if s[-1].isalpha():
        s += "."
    return s


def preprocess_result(res, allowed_metrics=None):
    out = dict()
    agg_metrics = res.aggregate_metrics
    for key, value in agg_metrics.items():
        if allowed_metrics is None or key in allowed_metrics:
            out[key] = value
    return out


def apply_sfr_template(query: str) -> str:
    # from https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
    task_description = "Retrieve semantically similar text."
    return f"Instruct: {task_description}\nQuery: {query}"


# def apply_personalization_template(user_info_text: str) -> str:
#     """
#     Create a prompt for extracting key personalization features from user-related text.
    
#     The input text may include:
#     - User profile information (e.g., demographics, preferences, characteristics)
#     - User history (e.g., past actions, interactions, content, behavioral patterns)
#     - Or both
    
#     The goal is to guide the embedding model to focus on extracting features
#     that are relevant for personalization, preference modeling, and behavioral analysis.
#     """
#     task_description = (
#         "Extract and represent the core personalization features that captures the user's unique characteristics, "
#         "preferences, and behavioral patterns from the given user information. "
#         "Focus on demographics, stated preferences, implicit preferences, behavioral patterns, "
#         "interests, interaction tendencies, and other characteristics that could inform personalization."
#     )
#     return f"# Instruct: {task_description}\n\n# Task Description: {task_description}\n\n# User Information:\n{user_info_text.strip()}"

def apply_personalization_template(user_info_text: str, task_description: str = None) -> str:
    """
    Create a prompt for extracting task-relevant personalization features from user-related text.

    The input may include:
    - User profile (e.g., demographics, stated or inferred preferences)
    - User history (e.g., past interactions, behaviors, content patterns)
    - Or both

    The goal is to guide the embedding model to extract features useful for personalization,
    preference modeling, and behavioral understanding in task-specific settings.
    
    Args:
        user_info_text: The user profile or history information
        task_description: Optional description of the specific task for context-aware personalization
    """
    base_instruction = (
        "Extract and represent key personalization features that reflect the user's unique characteristics, "
        "preferences, and behavioral patterns from the provided user information. Focus on demographics, "
        "stated and implicit preferences, interaction history, behavioral trends, and other traits that can "
        "inform task-specific personalization."
    )
    
    if task_description:
        task_aware_instruction = (
            f"{base_instruction} Pay special attention to features relevant for the following task: {task_description}"
        )
        
        return f"# Instruct:\n{task_aware_instruction}\n\n# User Information:\n{user_info_text.strip()}"
    else:
        return f"# Instruct:\n{base_instruction}\n\n# User Information:\n{user_info_text.strip()}"


def format_profile_text(profile_text, user_profile_format: str = "history", profile_all_history: str = "", data_entry: dict = None, profile_k: int = 0, task_name: str = None, include_history_stat: bool = False) -> str:
    """
    Format profile text based on the specified format for LaMP and LongLaMP datasets.
    
    Args:
        profile_text: The generated user profile text (string)
        user_profile_format: Format to use ("history", "gen_profile", or "mix")
        profile_all_history: The user history text (string) - used when profile_k=0
        data_entry: The full data entry dict containing profile_retrieval_k{k} fields
        profile_k: Which user history to use (0 for profile_all_history, or 1,2,4,8,12,16 for profile_retrieval_k{k})
        task_name: The task name to determine if history stats should be included
        include_history_stat: Whether to include history statistics for lamp_movie and lamp_news_cat tasks
        
    Returns:
        Formatted profile text string
    """
    # Ensure profile_text is a string
    if not isinstance(profile_text, str):
        profile_text = str(profile_text) if profile_text is not None else ""
    
    # Determine which user history to use based on profile_k
    if profile_k == 0:
        # Use profile_all_history
        user_history = str(profile_all_history) if profile_all_history is not None else ""
    else:
        # Use profile_retrieval_k{profile_k}
        if profile_k not in [1, 2, 4, 8, 12, 16, 32]:
            raise ValueError(f"Invalid profile_k: {profile_k}. Must be 0 or one of [1,2,4,8,12,16,32]")
        
        profile_retrieval_key = f"profile_retrieval_k{profile_k}"
        if data_entry and profile_retrieval_key in data_entry:
            user_history = str(data_entry[profile_retrieval_key]) if data_entry[profile_retrieval_key] is not None else ""
        else:
            # Fallback to profile_all_history if specific retrieval field not found
            user_history = str(profile_all_history) if profile_all_history is not None else ""
    
    # For PRISM, EC, PersonalReddit, and ALOE tasks: use only profile_text, no user history
    if is_gen_profile_only_task(task_name):
        # Explicitly ignore any history for these tasks
        user_history = ""
        user_profile_format = "gen_profile"
    
    # Prepare the base formatted text
    if user_profile_format == "history":
        # Use user history
        formatted_text = user_history
    elif user_profile_format == "gen_profile":
        # Use generated profile
        formatted_text = profile_text
    elif user_profile_format == "mix":
        # Concatenate both with template
        if profile_text and user_history:
            # Check if both are the same (happens for profile_text_only datasets)
            if profile_text == user_history:
                # If they're the same, just return the profile_text to avoid redundancy
                formatted_text = f"## User Profile: {profile_text}"
            else:
                formatted_text = f"## User Profile: {profile_text}\n\n## User History: {user_history}"
        elif profile_text:
            formatted_text = f"## User Profile: {profile_text}"
        elif user_history:
            formatted_text = f"## User History: {user_history}"
        else:
            raise ValueError(f"No profile text or user history found for profile_k={profile_k}")
    else:
        raise ValueError(f"Invalid user_profile_format: {user_profile_format}. Must be 'history', 'gen_profile', or 'mix'")
    
    # For lamp_movie and lamp_news_cat tasks, include history statistics if enabled
    if include_history_stat and data_entry and "history_stat" in data_entry:
        history_stat = data_entry["history_stat"]
        if history_stat:  # Only add if history_stat is not empty
            formatted_text += f"\n\n## User History Statistics: {history_stat}"
    
    return formatted_text


def create_personalization_template_fn(user_profile_format: str = "history", task_description: str = None):
    """
    Create a personalization template function with the specified profile format and task description.
    
    Args:
        user_profile_format: Format to use for profile text ("history", "gen_profile", or "mix")
        task_description: Optional description of the specific task for context-aware personalization
        
    Returns:
        A function that formats profile text and applies personalization template
    """
    def personalization_template_fn(profile_text) -> str:
        formatted_text = format_profile_text(profile_text, user_profile_format)
        return apply_personalization_template(formatted_text, task_description)
    
    return personalization_template_fn


def _iter_template_paths(candidate: str):
    """Yield possible local paths for a chat template candidate."""
    if not isinstance(candidate, str):
        return

    candidate = candidate.strip()
    if not candidate:
        return

    seen = set()
    candidate_path = Path(candidate)

    def _yield(path: Path):
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        yield path

    if candidate_path.is_absolute():
        if candidate_path.is_dir():
            yield from _yield(candidate_path / "chat_template.jinja")
        yield from _yield(candidate_path)
        return

    normalized = Path(candidate.strip("/"))
    for base_dir in (CHAT_TEMPLATE_BASE_DIR, REPO_ROOT):
        potential = base_dir / normalized
        if potential.is_dir():
            yield from _yield(potential / "chat_template.jinja")
        yield from _yield(potential)


def _find_local_chat_template(metadata: dict, apply_chat_template_fn):
    """Locate a chat template file for the provided tokenizer metadata."""
    tokenizer = getattr(apply_chat_template_fn, "__self__", None)
    if tokenizer is None:
        return None

    candidates = []
    for key in ("chat_template_path", "chat_template_name", "chat_template"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    name_or_path = getattr(tokenizer, "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path.strip():
        model_id = name_or_path.strip()
        candidates.append(model_id)
        base_name = Path(model_id).name
        if base_name != model_id:
            candidates.append(base_name)

    for candidate in candidates:
        for path in _iter_template_paths(candidate):
            if path.is_file():
                return path

    return None


def _prepare_chat_template(metadata: dict, apply_chat_template_fn):
    """Ensure the tokenizer behind apply_chat_template_fn uses the local chat template."""
    tokenizer = getattr(apply_chat_template_fn, "__self__", None)
    if tokenizer is None:
        return apply_chat_template_fn

    template_path = _find_local_chat_template(metadata, apply_chat_template_fn)
    if template_path is None:
        return apply_chat_template_fn

    cached_path = getattr(tokenizer, "_local_chat_template_path", None)
    if cached_path == str(template_path):
        return tokenizer.apply_chat_template

    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError:
        return apply_chat_template_fn

    normalized_template = template_text.replace("    ", "").replace("\n", "")
    if getattr(tokenizer, "chat_template", None) != normalized_template:
        tokenizer.chat_template = normalized_template
    tokenizer._local_chat_template_path = str(template_path)
    return tokenizer.apply_chat_template


def get_prompt_formatting_fn(
    metadata,
    sft_mode: Literal["causal_lm", "completion"],
    apply_chat_template_fn: Callable,
    is_intx_model: bool,
):
    assert sft_mode in ["causal_lm", "completion"], f"Invalid training task: {sft_mode}"

    if is_intx_model:
        apply_chat_template_fn = _prepare_chat_template(metadata, apply_chat_template_fn)

    def f(example):
        output_texts = dict(text=[]) if sft_mode == "causal_lm" else dict(prompt=[], response=[])
        df = pd.DataFrame(dict(example))
        for i, inp_txt in df.iterrows():
            # Special handling for PRISM conversations with apply_chat_template
            if inp_txt.get("is_prism_conversation", False):
                conversation = list(inp_txt["conversation"])  # copy to avoid modifying original
                response = str(inp_txt["answer"])
                
                # Optionally prepend user profile based on metadata flag
                prepend_flag = bool(metadata.get("prepand_profile", False))
                profile_text = inp_txt.get("profile_text", None)
                if prepend_flag and isinstance(profile_text, str) and len(profile_text.strip()) > 0:
                    conversation = (
                        [{"role": "user", "content": f"User Profile:\n{profile_text.strip()}"}] + conversation
                    )
                
                if sft_mode == "causal_lm":
                    # Add the assistant's response to the conversation
                    full_conversation = conversation + [{"role": "assistant", "content": response}]
                    text = apply_chat_template_fn(full_conversation, tokenize=False, add_generation_prompt=False)
                    output_texts["text"].append(text)
                elif sft_mode == "completion":
                    # Use conversation as prompt, response as target
    
                    prompt = apply_chat_template_fn(conversation, tokenize=False, add_generation_prompt=True)
                    output_texts["prompt"].append(prompt)
                    output_texts["response"].append(response)
            else:
                # Standard template processing
                if sft_mode == "causal_lm":
                    text = metadata["text_template"].format(**inp_txt)
                    output_texts["text"].append(text)
                elif sft_mode == "completion":
                    prompt = metadata["user_prompt_template"].format(**inp_txt)
                    output_texts["prompt"].append(prompt)
                    output_texts["response"].append(str(inp_txt[metadata["response_field"]]))
        return output_texts

    def f_intx(example):
        output_texts = dict(text=[]) if sft_mode == "causal_lm" else dict(prompt=[], response=[])
        df = pd.DataFrame(dict(example))
        for i, inp_txt in df.iterrows():
            # Special handling for PRISM conversations with apply_chat_template
            if inp_txt.get("is_prism_conversation", False):
                conversation = list(inp_txt["conversation"])  # copy to avoid modifying original
                response = str(inp_txt["answer"])
                
                # Optionally prepend user profile based on metadata flag
                prepend_flag = bool(metadata.get("prepand_profile", False))
                profile_text = inp_txt.get("profile_text", None)
                if prepend_flag and isinstance(profile_text, str) and len(profile_text.strip()) > 0:
                    conversation = (
                        [{"role": "user", "content": f"User Profile:\n{profile_text.strip()}"}] + conversation
                    )
                
                if sft_mode == "causal_lm":
                    # Add the assistant's response to the conversation
                    full_conversation = conversation + [{"role": "assistant", "content": response}]
                    text = apply_chat_template_fn(full_conversation, tokenize=False, add_generation_prompt=False)
                    output_texts["text"].append(text)
                elif sft_mode == "completion":
                    # Use conversation as prompt, response as target
                    prompt = apply_chat_template_fn(conversation, tokenize=False, add_generation_prompt=True)
                    output_texts["prompt"].append(prompt)
                    output_texts["response"].append(response)
            else:
                # Standard template processing for instruction-tuned models
                # NOTE: we assume specific chat_template here
                # that the chat_template should not have a default system_message
                # and it skils the system header if system_message is not provided
                # that is, using apply_chat_template to response_chat would not add the system_message
                prompt_chat = [
                    {"role": "system", "content": metadata["system_message"].format(**inp_txt)},
                    {"role": "user", "content": metadata["user_prompt_template"].format(**inp_txt)},
                ]
                response_chat = [
                    {
                        "role": "assistant",
                        "content": metadata["assistant_prefill"].format(**inp_txt)
                        + str(inp_txt[metadata["response_field"]]),
                    }
                ]
                if "assistant_postfill" in metadata:
                    response_chat[0]["content"] += metadata["assistant_postfill"].format(**inp_txt)
                if sft_mode == "causal_lm":
                    text = apply_chat_template_fn(prompt_chat + response_chat, tokenize=False, add_generation_prompt=False)
                    output_texts["text"].append(text)
                elif sft_mode == "completion":
                    # print(prompt_chat)
                    # print(response_chat)
                    prompt = apply_chat_template_fn(prompt_chat, tokenize=False, add_generation_prompt=False)
                    response = apply_chat_template_fn(response_chat, tokenize=False, add_generation_prompt=False)
                    # print(prompt)
                    # print(response)
                    # input()
                    output_texts["prompt"].append(prompt)
                    output_texts["response"].append(response)
        return output_texts

    return f if not is_intx_model else f_intx
