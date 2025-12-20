export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY=""

# uv run python eval_model.py \
#     --provider openai \
#     --input_file ./eval_results/Qwen/Qwen2.5-7B-Instruct/base_model/prism_random_test_eval_results.json \
#     --output_file ./eval_results/Qwen/Qwen2.5-7B-Instruct/base_model/prism_random_test_eval_results_llm_judge.json

for task in PAG_personalreddit_ood_test PAG_personalreddit_random_test PAG_EC_ood_test PAG_EC_random_test; do
    echo "Evaluating ${task}"
    uv run python eval_model.py \
        --provider openai \
        --input_file ./eval_results/Qwen/Qwen2.5-7B-Instruct/base_model/${task}_eval_results_k1.json \
        --output_file ./eval_results/Qwen/Qwen2.5-7B-Instruct/base_model/${task}_eval_results_k1_llm_judge_pointwise_ref.json
done

# personalreddit_random_test personalreddit_ood_test EC_random_test EC_ood_test prism_random_test prism_ood_test aloe_random_test aloe_ood_test
#  EC_random_test EC_ood_test prism_random_test prism_ood_test aloe_random_test aloe_ood_test
# prism_random_test prism_ood_test aloe_random_test aloe_ood_test

# uv run python eval_model.py \
#     --provider openai \
#     --input_file ./train_outputs/sft/hyper_lora/20250902-231138_lRvRSuZJ/eval_results/prism_ood_test_eval_results.json \
#     --output_file ./train_outputs/sft/hyper_lora/20250902-231138_lRvRSuZJ/eval_results/prism_ood_test_eval_results_llm_judge.json

# uv run python eval_model.py \
#     --provider openai \
#     --input_file ./train_outputs/sft/hyper_lora/20250902-231138_lRvRSuZJ/eval_results/prism_random_test_eval_results.json \
#     --output_file ./train_outputs/sft/hyper_lora/20250902-231138_lRvRSuZJ/eval_results/prism_random_test_eval_results_llm_judge.json
