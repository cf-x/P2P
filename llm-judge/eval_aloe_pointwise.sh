export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY=""

# uv run python eval_model.py \
#     --provider openai \
#     --input_file ./eval_results/Qwen/Qwen2.5-7B-Instruct/base_model/prism_random_test_eval_results.json \
#     --output_file ./eval_results/Qwen/Qwen2.5-7B-Instruct/base_model/prism_random_test_eval_results_llm_judge.json

uv run python eval_model.py \
    --provider openai \
    --input_file ./train_outputs/sft/hyper_lora/20250916-020856_sOoWajc7/checkpoints/it_100000/eval_results/prism_ood_test_eval_results.json \
    --output_file ./train_outputs/sft/hyper_lora/20250916-020856_sOoWajc7/checkpoints/it_100000/eval_results/prism_ood_test_eval_results_llm_judge_pointwise.json

# uv run python eval_model.py \
#     --provider openai \
#     --input_file ./eval_results/Qwen/Qwen2.5-7B-Instruct/base_model/PAG_aloe_random_test_eval_results_k1.json \
#     --output_file ./eval_results/Qwen/Qwen2.5-7B-Instruct/base_model/PAG_aloe_random_test_eval_results_llm_judge_k1.json

# uv run python eval_model.py \
#     --provider openai \
#     --input_file ./train_outputs/sft/hyper_lora/20250902-231138_lRvRSuZJ/eval_results/prism_ood_test_eval_results.json \
#     --output_file ./train_outputs/sft/hyper_lora/20250902-231138_lRvRSuZJ/eval_results/prism_ood_test_eval_results_llm_judge.json

# uv run python eval_model.py \
#     --provider openai \
#     --input_file ./train_outputs/sft/hyper_lora/20250902-231138_lRvRSuZJ/eval_results/prism_random_test_eval_results.json \
#     --output_file ./train_outputs/sft/hyper_lora/20250902-231138_lRvRSuZJ/eval_results/prism_random_test_eval_results_llm_judge.json
