#!/bin/bash

VIRTUAL_ENV='~/text-to-lora/.venv/bin/activate'

uv run python create_hf_datasets.py \
    --output_dir hf_datasets_history_split_task_specific_profile_retrieval_v4_test \
    --include_generated_profile \
    --generated_profile_dir generated_profile_task_specific \
    --data_version v9 \
    --k 1 2 4 8 12 16 32 \    
