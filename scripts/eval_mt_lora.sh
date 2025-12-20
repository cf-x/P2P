#!/bin/bash

# Evaluate a multi-task LoRA baseline checkpoint and save results
# Results are written under the LoRA checkpoint directory: <lora_dir>/eval_results

CUDA_VISIBLE_DEVICES=7 uv run python scripts/eval_mt_lora_checkpoint.py \
    --lora_dir ./train_outputs/sft/mt_lora/20250926-231342_puwCTTBB \
    --tasks EC_random_test EC_ood_test aloe_ood_test aloe_random_test lamp_citation_random_test lamp_movie_random_test lamp_news_cat_random_test lamp_news_headline_random_test lamp_product_random_test lamp_scholarly_title_random_test lamp_tweet_random_test \
    longlamp_abstract_generation_random_test longlamp_product_review_random_test longlamp_topic_writing_random_test \
    lamp_citation_ood_test lamp_movie_ood_test lamp_news_cat_ood_test lamp_news_headline_ood_test lamp_product_ood_test lamp_scholarly_title_ood_test lamp_tweet_ood_test \
    longlamp_abstract_generation_ood_test longlamp_product_review_ood_test longlamp_topic_writing_ood_test \
    personalreddit_ood_test personalreddit_random_test

    # To run more tasks, uncomment and append to --tasks:
    # prism_random_test prism_ood_test 
    # lamp_citation_random_test lamp_movie_random_test lamp_news_cat_random_test lamp_news_headline_random_test lamp_product_random_test lamp_scholarly_title_random_test lamp_tweet_random_test \
    # longlamp_abstract_generation_random_test longlamp_product_review_random_test longlamp_topic_writing_random_test  \
    # lamp_citation_ood_test lamp_movie_ood_test lamp_news_cat_ood_test lamp_news_headline_ood_test lamp_product_ood_test lamp_scholarly_title_ood_test lamp_tweet_ood_test \
    # longlamp_abstract_generation_ood_test longlamp_product_review_ood_test longlamp_topic_writing_ood_test
