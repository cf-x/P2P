from typing import Literal, Set, TypeAlias, get_args

from pydantic import BaseModel


CSBenchDomain: TypeAlias = Literal[
    "Computer Network", "Computer Organization", "Data Structure and Algorithm", "Operating System"
]
CS_BENCH_DOMAINS: Set[CSBenchDomain] = set(get_args(CSBenchDomain))

CSBenchSubDomain: TypeAlias = Literal[
    "Application Layer",
    "Bus",
    "Central Processing Unit",
    "Data Link Layer",
    "Data Representation and Operation",
    "File Management",
    "Graph",
    "Input/Output Management",
    "Input/Output System",
    "Instruction System",
    "Linear List",
    "Memory Management",
    "Network Layer",
    "Overview and Architecture",
    "Overview",
    "Physical Layer",
    "Processes and Threads",
    "Searching",
    "Sorting",
    "Stack, Queue, and Array",
    "Storage System",
    "String",
    "Transport Layer",
    "Tree",
]
CS_BENCH_SUB_DOMAINS: Set[CSBenchSubDomain] = set(get_args(CSBenchSubDomain))

CSBenchFormat: TypeAlias = Literal[
    "Assertion",
    "Fill-in-the-blank",
    "Multiple-choice",
    "Open-ended",
]
CS_BENCH_FORMATS: Set[CSBenchFormat] = set(get_args(CSBenchFormat))

CSBenchTag: TypeAlias = Literal["Knowledge", "Reasoning"]
CS_BENCH_TAGS: Set[CSBenchTag] = set(get_args(CSBenchTag))


class CSBenchTaskConfig(BaseModel):
    """
    TODO: Support other kinds of eval method (e.g. exact match, probability based eval)
    for multiple choice and assertion.
    Currently it only support CS-Bench paper's eval method (regex matching with word boundary).
    """

    # only supports 0-shot and 5-shot as per paper implementation
    num_shots: Literal[0, 5] = 0

    # By default, fewshot examples will be extracted from valid dataset
    fewshot_hf_repo: str = "SakanaAI/CS-Bench"
    fewshot_hf_data_files: str = "strfied_en_valid.jsonl"
