VIRTUAL_ENV=~/text-to-lora/.venv
export OPENAI_API_KEY="EMPTY"

# Example training script with HyperLoRA
# To disable cleanup of generated LoRAs after evaluation (useful for debugging), add:
# --delete_generated_loras_after_eval=false

CUDA_VISIBLE_DEVICES=5 WANDB_MODE=disabled uv run python scripts/train_custom_sft.py \
    configs/hyper_lora_p13n_unified.yaml \
    --model_dir=Qwen/Qwen2.5-3B-Instruct \
    --emb_model=Qwen/Qwen3-Embedding-4B \
    --warmup_frac=0.2 --lr=2e-5 \
    --grad_accum_steps=8 \
    --epochs=1 \
    --exp_setup=hyper_lora --encoder_type=linear \
    --l2_reg_generated_w=1e-3 --label_smoothing=0.1 \
    --neftune_noise_alpha=5 --weight_decay=1e-2 \
    --val_batch_size=16 \
    --use_api_embedding=true \
    --vllm_api_base="http://localhost:8002" \
    --vllm_embedding_model="Qwen/Qwen3-Embedding-4B" \
    --vllm_api_key="EMPTY" \
    --n_tasks_per_batch=4 \
    --n_points_per_task=1 \
    --dataset_sampling_strategy=sqrt_size \
    --user_profile_format=mix \
    --profile_k=2 \
    --skip_val=true \
    --include_history_stat=true \
    --val_freq=10000 \
    --use_hierarchical_sampler=false \
    --exp_setup=mt_lora