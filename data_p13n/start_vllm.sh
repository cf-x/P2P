
# Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=4,5 \
vllm serve Qwen/Qwen2.5-7B-Instruct \
--api_key EMPTY \
--served-model-name Qwen2.5-7B-Instruct \
--tensor-parallel-size 2 \
--port 8002