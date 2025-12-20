#!/bin/bash

VIRTUAL_ENV='../.venv/bin/activate'

# Set the API base URL and key
export OPENAI_API_BASE="http://localhost:8002/v1"
export OPENAI_API_KEY="EMPTY"


uv run python generate_user_profiles.py \
    --lamp_dir ./LaMP \
    --longlamp_dir ./LongLaMP \
    --prism_dir ./PRISM \
    --output_dir ./generated_profile_task_specific_unified \
    --prompts_dir ./prompts \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --max_context_length 22000 \
    --batch_size 200 \
    --use_unified_template