VIRTUAL_ENV='../.venv/bin/activate'
uv run python generate_user_embeddings.py \
    --data_dir ./ \
    --output_dir ./user_gen_profile_embeddings_task_specific/ \
    --use_api \
    --api_model "Qwen/Qwen3-Embedding-4B" \
    --api_key "EMPTY" \
    --api_base http://localhost:8002 \
    --batch_size 100 \
    --api_max_tokens_per_text 25000 \
    --use_generated_profiles \
    --generated_profile_dir ./generated_profile_task_specific/
