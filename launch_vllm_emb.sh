sudo docker run --gpus '"device=0,1"' \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8002:8002 \
    --ipc=host \
    vllm/vllm-openai:v0.10.0 \
    --model Qwen/Qwen3-Embedding-4B \
    --tensor-parallel-size 2 \
    --api_key EMPTY \
    --port 8002 \
    --task embed 
   
# 
    # --max_model_len 32768 \
    # 
    #  --task embed 
