import asyncio
import random
import re
import os
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncAzureOpenAI, AsyncOpenAI
from jinja2 import Template


# ==========================
# PRISM helpers
# ==========================

def conversation_to_instruction(turns: List[Dict[str, str]]) -> str:
    """Format a multi-turn conversation (PRISM `user_input`) into a single instruction string.

    Each item has keys: `role` (e.g., "user" or "assistant") and `content`.
    We include all turns in order, each on its own paragraph, as "role: content".
    """
    lines: List[str] = []
    for t in turns:
        role = t.get("role", "user")
        content = t.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def sample_to_instruction(sample: Dict[str, Any]) -> str:
    """Resolve an instruction string from a single eval sample.

    Rules:
    - If `user_instruction` is a non-empty string, use it directly.
    - Else if `user_input` is a list of {role, content} (PRISM/ALOE), convert via conversation_to_instruction.
    - Else if `user_input` is a non-empty string, use it directly.
    - Else fall back to `last_user_input`, `problem`, or empty string.
    """
    if not isinstance(sample, dict):
        return ""

    ui = sample.get("user_instruction")
    if isinstance(ui, str) and ui.strip():
        return ui.strip()

    u = sample.get("user_input")
    if isinstance(u, list):
        return conversation_to_instruction(u)
    if isinstance(u, str) and u.strip():
        return u.strip()

    lui = sample.get("last_user_input")
    if isinstance(lui, str) and lui.strip():
        return lui.strip()

    prob = sample.get("problem") or sample.get("question")
    if isinstance(prob, str) and prob.strip():
        return prob.strip()

    return ""


def build_instruction_from_pair(primary: Dict[str, Any], secondary: Dict[str, Any]) -> str:
    """Resolve instruction using primary sample first, then secondary as fallback.

    Supports both PRISM/ALOE (list-of-dicts conversations) and non-conversation
    tasks where the instruction is already a string.
    """
    instr = sample_to_instruction(primary)
    if instr:
        return instr
    return sample_to_instruction(secondary)


def load_prompt_from_file(file_path: str) -> Template:
    with open(file_path, 'r', encoding='utf-8') as f:
        return Template(f.read())


def load_prism_samples(input_file: str, task_key: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load PRISM samples from an evaluation results JSON.

    Expected shapes:
    - { "<task_key>": [ { "path": str, "sampled_res_details": [...] }, ... ] }
    - or { "sampled_res_details": [...] }
    - or [ { "sampled_res_details": [...] }, ... ]

    Returns (samples, meta) where samples is the list from `sampled_res_details`, and meta
    includes best-effort fields like `task` and `path`.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta: Dict[str, Any] = {}

    # Case 1: explicit task key
    if isinstance(data, dict) and task_key and task_key in data:
        arr = data[task_key]
        if isinstance(arr, list) and arr:
            first = arr[0]
            meta["task"] = task_key
            meta["path"] = first.get("path") if isinstance(first, dict) else None
            if isinstance(first, dict) and "sampled_res_details" in first:
                return first["sampled_res_details"], meta

    # Case 2: dict with a sampled_res_details key at top level
    if isinstance(data, dict) and "sampled_res_details" in data:
        return data["sampled_res_details"], meta

    # Case 3: dict of lists under unknown key
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list) and v:
                first = v[0]
                if isinstance(first, dict) and "sampled_res_details" in first:
                    meta["task"] = k
                    meta["path"] = first.get("path")
                    return first["sampled_res_details"], meta

    # Case 4: list at the top level
    if isinstance(data, list) and data:
        for item in data:
            if isinstance(item, dict) and "sampled_res_details" in item:
                meta["path"] = item.get("path")
                return item["sampled_res_details"], meta

    raise ValueError(f"Could not find 'sampled_res_details' in {input_file}. Provide --task_key if needed.")





"""PRISM evaluator"""

def extract_score(text: str) -> Optional[int]:
    # First try to match the pattern with brackets: [RESULT] 1-5
    match = re.search(r'\[RESULT\]\s*([1-5])', text)
    if match:
        return int(match.group(1))

    # Try to match "[Result] integer" pattern
    match = re.search(r'\[Result\]\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to match "**[RESULT]** integer" pattern
    match = re.search(r'\*\*\[RESULT\]\*\*\s*([1-5])', text)
    if match:
        return int(match.group(1))

    # If no match, try the pattern without brackets: RESULT 1-5
    match = re.search(r'RESULT:\s+([1-5])', text)
    if match:
        return int(match.group(1))

    # Try to match "**Result**: integer" pattern
    match = re.search(r'\*\*Result\*\*:\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to match "**Result:** integer" pattern
    match = re.search(r'\*\*Result:\*\*\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to match "**Score**: integer" pattern
    match = re.search(r'\*\*Score\*\*:\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to match "**Score:** integer" pattern
    match = re.search(r'\*\*Score:\*\*\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to match "Score: integer" pattern
    match = re.search(r'Score:\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to match "Result: integer" pattern
    match = re.search(r'Result:\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to match "Score: **integer**" pattern
    match = re.search(r'Score:\s*\*\*([1-5])\*\*', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to match "Score integer" pattern (without colon)
    match = re.search(r'Score\s+([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # ALoE often returns only a single digit (1-5)
    stripped = text.strip()
    if re.fullmatch(r'[1-5]', stripped):
        return int(stripped)

    # Fallback: find the last standalone digit 1-5 in text
    matches = list(re.finditer(r'(?:^|\D)([1-5])(?=\D|$)', text))
    if matches:
        return int(matches[-1].group(1))

    # If no score is found, return None
    return None

import re

def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_pairwise_winner(text: str) -> Optional[str]:
    """Extract the pairwise winner label ('A', 'B', or 'C') from judge output.

    Updated to support the prompt in llm_eval_prompt_pairwise.md which asks for
    a final verdict strictly in this format: "[[A]]", "[[B]]", or "[[C]]" (tie).

    Returns 'A', 'B', or 'C' (uppercase) if confidently found, else None.
    """
    if not isinstance(text, str):
        return None

    s = text.strip()
    # 1) New primary format: [[A]] / [[B]] / [[C]] (case-insensitive)
    m = re.search(r"\[\[\s*([ABCabc])\s*\]\]", s)
    if m:
        return m.group(1).upper()

    # 2) Backward-compat: older cache prompt like "[RESULT] A" or variants
    m = re.search(r"\[RESULT\]\s*([ABC])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:Result|Winner)\s*:?\s*([ABC])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3) Very weak fallback: last standalone token A/B/C.
    #    Use cautiously to avoid accidental picks from the analysis text.
    m_all = list(re.finditer(r"(?:^|\W)([ABC])(?:\W|$)", s, flags=re.IGNORECASE))
    if m_all:
        return m_all[-1].group(1).upper()
    return None


async def call_llm_async(
    client: Any,
    provider: str,
    messages: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    custom_id: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    is_json: bool = False,
    is_reason: bool = False,
) -> None:
    try:
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if is_json:
            kwargs["response_format"] = {"type": "json_object"}
        if not is_reason:
            kwargs["temperature"] = temperature
            if provider == "openai":
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        response = await client.chat.completions.create(**kwargs)
        _body = response.model_dump_json()
        results.append({"custom_id": custom_id, "response": {"body": _body}})

    except Exception as e:
        print(f"Task {custom_id} failed: {e}. Retrying in 5-30 seconds...")
        await asyncio.sleep(random.randint(10, 30))
        try:
            response = await client.chat.completions.create(**kwargs)  # type: ignore[name-defined]
            _body = response.model_dump_json()
            results.append({"custom_id": custom_id, "response": {"body": _body}})
        except Exception as e2:
            print(f"Task {custom_id} failed again: {e2}. Skipping...")
    finally:
        await asyncio.sleep(random.uniform(0, 10))


async def call_llm_in_parallel(
    client: Any,
    provider: str,
    requests: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 1.0,
    max_tokens: int = 10000,
    output_path: Optional[str] = None,
    batch_size: int = 10,
    is_json: bool = False,
    is_reason: bool = False,
) -> List[Dict[str, Any]]:
    num_calls = len(requests)
    results: List[Dict[str, Any]] = []
    batch_results: List[Dict[str, Any]] = []
    print("Starting LLM calls...")

    for i in range(0, num_calls, batch_size):
        batch_tasks: List[asyncio.Task] = []
        for j in range(batch_size):
            if i + j < num_calls:
                _request = requests[i + j]
                _messages = _request['body']['messages']
                _custom_id = _request['custom_id']
                # Overwrite model_name and temperature if provided per-request
                _model_name = (
                    _request['body'].get('model') if model_name is None else model_name
                )
                _temperature = _request['body'].get('temperature', temperature)
                batch_tasks.append(
                    call_llm_async(
                        client=client,
                        provider=provider,
                        messages=_messages,
                        results=batch_results,
                        custom_id=_custom_id,
                        model_name=_model_name,
                        temperature=_temperature,
                        max_tokens=max_tokens,
                        is_json=is_json,
                        is_reason=is_reason,
                    )
                )
        await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        if output_path:
            with open(output_path, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    f.write(json.dumps(result) + '\n')
            print(
                f"Batch {i // batch_size + 1} completed and saved to file. Total tasks completed: {i + len(batch_tasks)} / {num_calls}",
                flush=True,
            )
        else:
            print(
                f"Batch {i // batch_size + 1} completed. Total tasks completed: {i + len(batch_tasks)} / {num_calls}",
                flush=True,
            )
        batch_results.clear()

    print("All tasks completed.")
    return results

# _endpoint = "https://roar-dev-eastus.openai.azure.com/" #"https://oar-oai.openai.azure.com/" # https://oar-oai-swedencentral.openai.azure.com/
# region = 'roar-dev-eastus'
# _key = keyvault.get_secret(region) # "aoai-east-us"

async def main(
    provider: str,
    # Azure config
    azure_endpoint: Optional[str] = None,
    azure_api_version: str = "2024-12-01-preview",
    azure_key: Optional[str] = None,
    azure_secret_name: Optional[str] = None,
    region: Optional[str] = None,
    # OpenAI/vLLM config
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    # Checkpointing config (legacy)
    # PRISM inputs/outputs
    input_file: Optional[str] = None,
    task_key: Optional[str] = None,
    output_file: Optional[str] = None,
    # Evaluation mode
    mode: str = "pointwise",
    # Optional second input for pairwise comparison
    input_file_b: Optional[str] = None,
    task_key_b: Optional[str] = None,
) -> None:
    # Choose evaluation model
    if provider == "azure":
        if azure_endpoint == "https://roar-dev-northcentralus.openai.azure.com/":
            eval_model_name = "gpt-4o-data-zone"
        else:
            eval_model_name = "gpt-4o-0806"
    else:
        # Reasonable OpenAI default; can be overridden in requests
        eval_model_name = "gpt-4o-2024-08-06"

    # Mode A: PRISM/ALoE evaluation from an input file
    prism_mode = input_file is not None
    requests: List[Dict[str, Any]] = []
    sample_index_by_id: Dict[str, int] = {}
    samples: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    pair_assignment: Dict[str, Dict[str, str]] = {}
    # pair_assignment[custom_id] = { 'prompt_A_from': 'model_a'|'model_b', 'prompt_B_from': 'model_a'|'model_b', 'qid': qid }

    if prism_mode:
        # Load samples (works for PRISM and ALoE file shapes)
        samples, meta = load_prism_samples(input_file=input_file, task_key=task_key)

        # Detect ALoE vs PRISM by task key or file name
        task_name = (meta.get("task") or "").lower()
        aloe_mode = ("aloe" in task_name) or (input_file and ("aloe" in os.path.basename(input_file).lower()))

        # Prompt templates
        eval_prompt_template = load_prompt_from_file('llm_eval_prompt.md')
        eval_prompt_pairwise = load_prompt_from_file('llm_eval_prompt_pairwise.md')
        score_rubric_template = load_prompt_from_file('score_rubric.md')

        # Define separate rubrics for PRISM and ALoE. Only ALoE rubric is modified per request.
        # PRISM rubric (unchanged): personalization to profile/personality, content relevance, human-likeness.
        prism_rubric_data = {
            "criteria": "Evaluate how well the response to the instruction is personalized to the specific user.",
            "score1_description": (
                "Generic or impersonal. Ignores the provided profile/personality. Style does not match the user; may feel "
                "robotic or off-topic. Makes incorrect assumptions or contradicts stated preferences. No meaningful use of "
                "user details; largely boilerplate."
            ),
            "score2_description": (
                "Minimal personalization. Mentions a profile detail superficially but remains mostly generic. Weak style "
                "match; limited relevance to the user's interests or situation. Includes filler or distracting disclaimers. "
                "Significant deviation from the reference's intent or emphasis."
            ),
            "score3_description": (
                "Basic personalization. References a few relevant details and partially adapts tone. Generally on topic but "
                "misses important user nuances (interests, constraints, or personality cues). Moderate similarity to the "
                "reference; may be verbose or somewhat generic."
            ),
            "score4_description": (
                "Good personalization. Integrates multiple user details accurately; content is relevant and helpful. Tone "
                "largely matches the user's personality and preferred style. Clear, concise, and engaging with only minor "
                "misses versus the user's preferences or the reference's intent."
            ),
            "score5_description": (
                "Excellent personalization. Seamlessly weaves in pertinent profile details; highly relevant and tailored "
                "guidance or conversation. Tone precisely matches the user's personality—empathetic, engaging, and concise. "
                "Avoids boilerplate and unnecessary disclaimers. Closely aligned with the user's likely preference as "
                "indicated by the reference."
            ),
        }

        # ALoE rubric (modified): focus strictly on three personalization aspects requested.
        # aloe_rubric_data = {
        #     "criteria": (
        #         "Evaluate how well the response is personalized to the specific user given the profile, personality, and instruction. "
        #         "Consider but not limited to three axes: (1) Style alignment to the user's personality and preferred tone; (2) Content "
        #         "relevance and specificity to the user's background, interests, needs, and constraints; (3) Human-likeness: "
        #         "engaging, empathetic, and concise while staying on topic. Use the reference answer as a signal of the "
        #         "user's preferred content/style, but prioritize faithfulness to the provided profile/personality."
        #     ),
        #     "score1_description": (
        #         "Not personalized. Style clearly mismatched to the user's personalities; content is generic or off-topic relative to the profile. "
        #         "Reads robotic or includes filler; not engaging; often verbose or meandering."
        #     ),
        #     "score2_description": (
        #         "Slight personalization. May mention a user detail but style remains poorly matched; relevance is weak or superficial. "
        #         "Delivery lacks a human touch and/or is not concise."
        #     ),
        #     "score3_description": (
        #         "Moderate personalization. Some tone adaptation and several relevant points from the profile. "
        #         "Generally understandable but still somewhat generic or wordy; engagement is uneven."
        #     ),
        #     "score4_description": (
        #         "Strong personalization. Tone mostly matches the user's personalities; integrates multiple profile-relevant details. "
        #         "Feels human and engaging; clear and succinct with only minor lapses."
        #     ),
        #     "score5_description": (
        #         "Excellent personalization. Tone is spot-on for the user's personalities; content tightly aligned with the profile. "
        #         "Natural, engaging, and concise throughout with no unnecessary filler or boilerplate."
        #     ),
        # }
        aloe_rubric_data = prism_rubric_data
        

        def parse_aloe_profile(text: str) -> Tuple[str, str]:
            """Extract user_profile and user_personality from a combined ALoE profile string.
            Returns (profile, personality). Falls back gracefully if not found.
            """
            if not isinstance(text, str):
                return "", ""
            # Normalize line endings
            s = text.strip()
            prof = ""
            pers = ""
            # Try to extract quoted content after headings
            m_prof = re.search(r"##\s*User\s*Profile:\s*\"?(.*?)\"?(?:\n|$)", s, re.IGNORECASE | re.DOTALL)
            if m_prof:
                prof = m_prof.group(1).strip()
            m_pers = re.search(r"##\s*User\s*Personality:\s*\"?(.*?)\"?(?:\n|$)", s, re.IGNORECASE | re.DOTALL)
            if m_pers:
                pers = m_pers.group(1).strip()
            # If still empty, attempt split heuristic
            if not prof and "User Profile:" in s:
                chunk = s.split("User Profile:", 1)[-1]
                if "User Personality:" in chunk:
                    prof = chunk.split("User Personality:", 1)[0].strip().strip('"')
                else:
                    prof = chunk.strip().strip('"')
            if not pers and "User Personality:" in s:
                pers = s.split("User Personality:", 1)[-1].strip().strip('"')
            return prof, pers

        if mode == "pairwise":
            # Load the second result file required for pairwise
            if not input_file_b:
                raise SystemExit("--input_file_b is required when --mode pairwise")
            samples_b, meta_b = load_prism_samples(input_file=input_file_b, task_key=task_key_b)
            meta_b_path = meta_b.get("path")

            # Build index by question_id for both A and B
            by_qid_a: Dict[str, Dict[str, Any]] = {}
            by_qid_b: Dict[str, Dict[str, Any]] = {}
            for idx, s in enumerate(samples):
                qid = s.get("question_id", f"idx_A_{idx}")
                by_qid_a[qid] = s
            for idx, s in enumerate(samples_b):
                qid = s.get("question_id", f"idx_B_{idx}")
                by_qid_b[qid] = s

            # Intersect qids present in both
            common_qids = [q for q in by_qid_a.keys() if q in by_qid_b]
            if not common_qids:
                raise SystemExit("No overlapping question_id between input files for pairwise comparison")

            # Compose pairwise requests with randomized A/B order
            for qid in common_qids:
                s_a = by_qid_a[qid]
                s_b = by_qid_b[qid]

                # Extract shared fields (instruction/user profile/reference)
                reference_answer = s_a.get("reference", "") or s_b.get("reference", "")
                raw_profile = s_a.get("user_profile", "") or s_b.get("user_profile", "")
                # Build instruction (PRISM/ALOE use conversation conversion; others may be strings)
                instruction = build_instruction_from_pair(s_a, s_b)

                # Build user_profile text and rubric
                if aloe_mode:
                    prof, pers = parse_aloe_profile(raw_profile)
                    combined_profile = f"User Profile: {prof}" if prof else ""
                    if pers:
                        combined_profile = (combined_profile + ("\n\n" if combined_profile else "")) + f"User Personality: {pers}"
                    rubric = score_rubric_template.render(**aloe_rubric_data)
                    user_profile_text = combined_profile
                else:
                    rubric = score_rubric_template.render(**prism_rubric_data) if score_rubric_template else ""
                    user_profile_text = raw_profile

                # Responses from the two systems
                resp_model_a = s_a.get("output", "")
                resp_model_b = s_b.get("output", "")

                # Randomize assignment to Response A / Response B to mitigate position bias
                if random.random() < 0.5:
                    response_A = resp_model_a
                    response_B = resp_model_b
                    assignment = {"prompt_A_from": "model_a", "prompt_B_from": "model_b"}
                else:
                    response_A = resp_model_b
                    response_B = resp_model_a
                    assignment = {"prompt_A_from": "model_b", "prompt_B_from": "model_a"}

                prompt = eval_prompt_pairwise.render(
                    instruction=instruction,
                    response_A=response_A,
                    response_B=response_B,
                    reference_answer=reference_answer,
                    user_profile=user_profile_text,
                    rubric=rubric,
                )
                # print(prompt)
                # print("--------------------------------"*5)
                messages = [{"role": "user", "content": prompt}]
                custom_id = qid  # keep qid as id; mapping is stored separately
                example = {
                    "custom_id": custom_id,
                    "body": {
                        "model": eval_model_name,
                        "messages": messages,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    },
                }

                # Track indices and assignment for later decoding
                sample_index_by_id[custom_id] = len(requests)
                pair_assignment[custom_id] = {**assignment, "qid": qid}
                requests.append(example)

            print("# PRISM pairwise requests:", len(requests))

            # Where to store raw judge completions (JSONL) and structured results (JSON)
            if output_file is None:
                base_a = os.path.splitext(os.path.basename(input_file))[0]
                base_b = os.path.splitext(os.path.basename(input_file_b))[0]
                output_dir = os.path.dirname(input_file)
                output_file = os.path.join(output_dir, f"{base_a}_VS_{base_b}.llm_judge_pairwise.json")
            checkpoint_file = os.path.splitext(output_file)[0] + ".raw.jsonl"
            try:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
            except Exception:
                pass
        else:
            # Pointwise: single-system evaluation
            for idx, s in enumerate(samples):
                question_id = s.get("question_id", f"idx_{idx}")
                response_eval = s.get("output", "")
                reference_answer = s.get("reference", "")
                raw_profile = s.get("user_profile", "")
                if aloe_mode:
                    # Instruction may be conversation (list) or direct string
                    instruction = sample_to_instruction(s)
                    prof, pers = parse_aloe_profile(raw_profile)
                    # Combine profile + personality into one profile string for the shared template
                    combined_profile = f"User Profile: {prof}" if prof else ""
                    if pers:
                        combined_profile = (combined_profile + ("\n\n" if combined_profile else "")) + f"User Personality: {pers}"
                    rubric = score_rubric_template.render(**aloe_rubric_data)
                    prompt = eval_prompt_template.render(
                        instruction=instruction,
                        response=response_eval,
                        reference_answer=reference_answer,
                        user_profile=combined_profile,
                        rubric=rubric,
                    )
                else:
                    instruction = sample_to_instruction(s)
                    rubric = score_rubric_template.render(**prism_rubric_data) if score_rubric_template else ""
                    prompt = eval_prompt_template.render(
                        instruction=instruction,
                        response=response_eval,
                        reference_answer=reference_answer,
                        user_profile=raw_profile,
                        rubric=rubric,
                    )

                messages = [{"role": "user", "content": prompt}]
                example = {
                    "custom_id": question_id,
                    "body": {
                        "model": eval_model_name,
                        "messages": messages,
                        "temperature": 0.1,
                        "top_p": 0.9,
                    },
                }

                sample_index_by_id[question_id] = idx
                requests.append(example)

            print("# PRISM requests:", len(requests))

            # Where to store raw judge completions (JSONL) and structured results (JSON)
            if output_file is None:
                base = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(os.path.dirname(input_file), f"{base}.llm_judge.json")
            checkpoint_file = os.path.splitext(output_file)[0] + ".raw.jsonl"
            # Ensure a clean raw output file to avoid mixing past runs
            try:
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
            except Exception:
                pass

    else:
        raise SystemExit("--input_file is required for PRISM evaluation")

    start = time.time()

    if provider == "azure":
        # Resolve Azure key
        resolved_key = azure_key
        if resolved_key is None:
            try:
                from azureml.core import Workspace  # type: ignore
                ws = Workspace.from_config()
                keyvault = ws.get_default_keyvault()
                secret_name = azure_secret_name or region
                if not secret_name:
                    raise RuntimeError("Provide --azure_key or --azure_secret_name/--region to load from Azure Key Vault")
                resolved_key = keyvault.get_secret(secret_name)
            except Exception as e:
                raise SystemExit(
                    f"Azure key not provided and Key Vault resolution failed: {e}. Pass --azure_key or configure azureml Workspace.from_config()."
                )
        if not azure_endpoint:
            raise SystemExit("--endpoint is required for provider=azure")
        async with AsyncAzureOpenAI(azure_endpoint=azure_endpoint, api_key=resolved_key, api_version=azure_api_version) as client:
            await call_llm_in_parallel(
                client=client,
                provider="azure",
                requests=requests,
                model_name=eval_model_name,
                output_path=checkpoint_file,
                batch_size=50,
                is_json=False,
                is_reason=False,
            )
    else:
        # OpenAI (or OpenAI-compatible)
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_api_base = api_base or os.environ.get("OPENAI_API_BASE")
        if not resolved_api_key:
            print("Warning: OPENAI_API_KEY not set; pass --api_key or export env var.")
        if resolved_api_base:
            client = AsyncOpenAI(api_key=resolved_api_key, base_url=resolved_api_base)
        else:
            client = AsyncOpenAI(api_key=resolved_api_key)
        await call_llm_in_parallel(
            client=client,
            provider="openai",
            requests=requests,
            model_name=eval_model_name,
            output_path=checkpoint_file,
            batch_size=50,
            is_json=False,
            is_reason=False,
        )

    end = time.time()
    interval = end - start
    print("used time: {}".format(interval))

    id2llm_feedback: Dict[str, str] = {}
    id2llmeval: Dict[str, int] = {}
    score_list: List[int] = []
    id2winner_model: Dict[str, str] = {}  # for pairwise: 'model_a' or 'model_b' or 'tie'
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            i = json.loads(line)
            output = json.loads(i['response']['body'])['choices'][0]['message']['content']
            id_ = i['custom_id']
            id2llm_feedback[id_] = output
            if mode == "pairwise":
                winner = extract_pairwise_winner(output)
                if not winner:
                    print(f"Winner extraction failed for index {idx} | {id_}: {output}")
                    id2winner_model[id_] = "tie"
                    continue
                assign = pair_assignment.get(id_, {})
                # Map 'A'/'B' to original model slot, 'C' means tie
                if winner == 'A':
                    prompt_from_key = 'prompt_A_from'
                    model_from = assign.get(prompt_from_key)
                    id2winner_model[id_] = model_from if model_from in ("model_a", "model_b") else "tie"
                elif winner == 'B':
                    prompt_from_key = 'prompt_B_from'
                    model_from = assign.get(prompt_from_key)
                    id2winner_model[id_] = model_from if model_from in ("model_a", "model_b") else "tie"
                else:
                    # 'C' or any other unexpected but non-empty marker => tie
                    id2winner_model[id_] = "tie"
            else:
                score = extract_score(output)
                if score is None:
                    print(f"Score extraction failed for index {idx} | {id_}: {output}")
                    continue
                id2llmeval[id_] = score
                score_list.append(score)

    import numpy as np
    score_arr = np.array(score_list) if score_list else np.array([0])

            # Save structured results for PRISM mode
    if prism_mode:
        if mode == "pairwise":
            wins_a = sum(1 for k, v in id2winner_model.items() if v == "model_a")
            wins_b = sum(1 for k, v in id2winner_model.items() if v == "model_b")
            ties = sum(1 for k, v in id2winner_model.items() if v == "tie")

            # Compose details per qid using available sample fields
            details: List[Dict[str, Any]] = []
            for qid, assign in pair_assignment.items():
                # There might be qids not present if request failed—guard with get
                feedback = id2llm_feedback.get(qid)
                winner_model = id2winner_model.get(qid)
                s_a = None
                s_b = None
                # Try reconstruct via first samples list (A) for basic metadata; we don't reload B list here
                # Because we didn't keep full mapping here, skip including full responses to keep minimal
                details.append({
                    "question_id": qid,
                    "prompt_A_from": assign.get("prompt_A_from"),
                    "prompt_B_from": assign.get("prompt_B_from"),
                    "judge_feedback": feedback,
                    "winner": winner_model,
                })

            # Compute win rates (ratios) and percentages
            total_comparisons = max(1, (wins_a + wins_b + ties))
            total_no_ties = max(1, (wins_a + wins_b))
            win_rate_a = float(wins_a / total_comparisons)
            win_rate_b = float(wins_b / total_comparisons)
            win_rate_a_excl_ties = float(wins_a / total_no_ties)
            win_rate_b_excl_ties = float(wins_b / total_no_ties)

            structured: Dict[str, Any] = {
                "task": meta.get("task"),
                "model_a_path": meta.get("path"),
                "model_b_path": meta_b_path if 'meta_b_path' in locals() else ((input_file_b and os.path.basename(input_file_b)) or None),
                "judge_model": eval_model_name,
                "count": len(pair_assignment),
                "wins_model_a": wins_a,
                "wins_model_b": wins_b,
                "ties": ties,
                # Ratios in [0,1] including ties in denominator
                "win_rate_model_a": win_rate_a,
                "win_rate_model_b": win_rate_b,
                # Ratios in [0,1] excluding ties from denominator
                "win_rate_model_a_excl_ties": win_rate_a_excl_ties,
                "win_rate_model_b_excl_ties": win_rate_b_excl_ties,
                # Percent values for convenience
                "win_rate_pct_model_a": round(win_rate_a * 100.0, 4),
                "win_rate_pct_model_b": round(win_rate_b * 100.0, 4),
                "win_rate_pct_model_a_excl_ties": round(win_rate_a_excl_ties * 100.0, 4),
                "win_rate_pct_model_b_excl_ties": round(win_rate_b_excl_ties * 100.0, 4),
                "details": details,
            }

            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(structured, f_out, ensure_ascii=False, indent=2)
            print(f"Saved pairwise LLM-as-a-judge results to: {output_file}")
        else:
            structured: Dict[str, Any] = {
                "task": meta.get("task"),
                "model_path": meta.get("path"),
                "judge_model": eval_model_name,
                "count": len(samples),
                "average_score": float(np.mean(score_arr)),
                "details": [],
            }
            for idx, s in enumerate(samples):
                qid = s.get("question_id", f"idx_{idx}")
                if aloe_mode:
                    prof, pers = parse_aloe_profile(s.get("user_profile", ""))
                    instruction = sample_to_instruction(s)
                    item = {
                        "question_id": qid,
                        "user_id": s.get("user_id"),
                        "instruction": instruction,
                        "response": s.get("output", ""),
                        "reference": s.get("reference", ""),
                        "user_profile": prof,
                        "user_personality": pers,
                        "judge_feedback": id2llm_feedback.get(qid),
                        "score": id2llmeval.get(qid),
                    }
                else:
                    item = {
                        "question_id": qid,
                        "user_id": s.get("user_id"),
                        "instruction": sample_to_instruction(s),
                        "response": s.get("output", ""),
                        "reference": s.get("reference", ""),
                        "user_profile": s.get("user_profile", ""),
                        "judge_feedback": id2llm_feedback.get(qid),
                        "score": id2llmeval.get(qid),
                    }
                structured["details"].append(item)

            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(structured, f_out, ensure_ascii=False, indent=2)
            print(f"Saved LLM-as-a-judge results to: {output_file}")

        # Print summary
        if mode == "pairwise":
            print(f"{eval_model_name} | Pairwise wins: A={wins_a}, B={wins_b}, ties={ties}")
            # Also print percentage summaries for quick viewing
            try:
                print(f"Win rate (incl. ties): A={structured['win_rate_pct_model_a']}%, B={structured['win_rate_pct_model_b']}%")
                print(f"Win rate (excl. ties): A={structured['win_rate_pct_model_a_excl_ties']}%, B={structured['win_rate_pct_model_b_excl_ties']}%")
            except Exception:
                pass
        else:
            print(f"{eval_model_name} | LLM Eval mean score: {np.mean(score_arr)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-judge evaluation. Supports PRISM inputs and legacy checkpoints.")

    parser.add_argument("--provider", type=str, choices=["azure", "openai"], default="azure", help="Backend provider for evaluation")

    # Azure options
    parser.add_argument("--endpoint", type=str, default="https://roar-dev-swedencentral.openai.azure.com/", help="Azure endpoint URL")
    parser.add_argument("--region", type=str, default="roar-dev-swedencentral", help="Region/secret name for Azure Key Vault fallback")
    parser.add_argument("--azure_api_version", type=str, default="2024-12-01-preview", help="Azure OpenAI API version")
    parser.add_argument("--azure_key", type=str, default=None, help="Azure API key (optional if Key Vault is configured)")
    parser.add_argument("--azure_secret_name", type=str, default=None, help="Secret name to fetch from Azure Key Vault if azure_key not provided")

    # OpenAI options
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for OpenAI-compatible endpoint (optional)")
    parser.add_argument("--api_key", type=str, default=None, help="API key for OpenAI (falls back to OPENAI_API_KEY)")

    # Checkpointing config
    parser.add_argument("--output_dir", type=str, default="eval_model_ckpt", help="Root directory for checkpoints")
    parser.add_argument("--run_id", type=str, default=None, help="Optional subdirectory to separate runs (e.g., timestamp or tag)")
    parser.add_argument("--checkpoint_provider", type=str, default=None, help="Provider directory to load checkpoint from (defaults to --provider)")

    # PRISM inputs/outputs
    parser.add_argument("--input_file", type=str, default=None, help="Path to PRISM eval_results JSON (e.g., prism_ood_test_eval_results.json)")
    parser.add_argument("--task_key", type=str, default=None, help="Top-level task key (e.g., prism_ood_test) if needed")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save structured LLM-judge results JSON")
    
    # Mode control and pairwise inputs
    parser.add_argument("--mode", type=str, choices=["pointwise", "pairwise"], default="pointwise", help="Evaluation mode: pointwise scoring or pairwise comparison")
    parser.add_argument("--input_file_b", type=str, default=None, help="Second eval_results JSON for pairwise comparison")
    parser.add_argument("--task_key_b", type=str, default=None, help="Top-level task key for second file if needed")

    args = parser.parse_args()

    asyncio.run(
        main(
            provider=args.provider,
            azure_endpoint=args.endpoint,
            azure_api_version=args.azure_api_version,
            azure_key=args.azure_key,
            azure_secret_name=args.azure_secret_name,
            region=args.region,
            api_base=args.api_base,
            api_key=args.api_key,
            input_file=args.input_file,
            task_key=args.task_key,
            output_file=args.output_file,
            mode=args.mode,
            input_file_b=args.input_file_b,
            task_key_b=args.task_key_b,
        )
    )
