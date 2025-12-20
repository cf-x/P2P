#!/bin/bash

# uv run python scripts/run_eval.py --model-dir  mistralai/Mistral-7B-Instruct-v0.2 --save-to-base-model-dir --tasks boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k humaneval mbpp
# uv run python scripts/run_eval.py --model-dir  meta-llama/Llama-3.1-8B-Instruct --save-to-base-model-dir --tasks boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k humaneval mbpp
# uv run python scripts/run_eval.py --model-dir  google/gemma-2-2b-it --use-icl --save-to-base-model-dir --tasks boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k humaneval mbpp
# export NUMBA_DISABLE_JIT=1 
CUDA_VISIBLE_DEVICES=7 uv run python scripts/run_eval.py \
    --model-dir Qwen/Qwen2.5-3B-Instruct \
    --save-to-base-model-dir \
    --tasks personalreddit_random_test personalreddit_ood_test EC_random_test EC_ood_test prism_random_test prism_ood_test aloe_random_test aloe_ood_test \
    lamp_citation_random_test lamp_movie_random_test lamp_news_cat_random_test lamp_news_headline_random_test lamp_product_random_test lamp_scholarly_title_random_test lamp_tweet_random_test longlamp_abstract_generation_random_test longlamp_product_review_random_test longlamp_topic_writing_random_test opinionqa_random_test \
    lamp_citation_ood_test lamp_movie_ood_test lamp_news_cat_ood_test lamp_news_headline_ood_test lamp_product_ood_test lamp_scholarly_title_ood_test lamp_tweet_ood_test longlamp_abstract_generation_ood_test longlamp_product_review_ood_test longlamp_topic_writing_ood_test opinionqa_ood_test
    
    # prism_random_test prism_ood_test aloe_random_test aloe_ood_test#    
    # --max-context-length 3000 \
