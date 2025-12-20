#!/bin/bash

CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_eval.py \
    --model-dir Qwen/Qwen2.5-3B-Instruct \
    --save-to-base-model-dir \
    --max-context-length 32768 \
    --retrieval-k 1 \
    --tasks PAG_lamp_movie_ood_test PAG_lamp_movie_random_test PAG_lamp_news_cat_ood_test PAG_lamp_news_cat_random_test PAG_lamp_citation_ood_test PAG_lamp_citation_random_test PAG_longlamp_abstract_generation_ood_test PAG_longlamp_abstract_generation_random_test \
    PAG_lamp_news_headline_ood_test PAG_lamp_news_headline_random_test PAG_lamp_product_ood_test PAG_lamp_product_random_test PAG_lamp_scholarly_title_ood_test PAG_lamp_scholarly_title_random_test PAG_lamp_tweet_ood_test PAG_lamp_tweet_random_test PAG_longlamp_product_review_ood_test \
    PAG_longlamp_product_review_random_test PAG_longlamp_topic_writing_ood_test PAG_longlamp_topic_writing_random_test
    
    # --use-yarn \
    # --yarn-scaling-factor 4.0 \
    # --yarn-original-max-position-embeddings 32768 \
    
    #  lamp_movie_random_test lamp_news_cat_random_test lamp_product_random_test opinionqa_random_test lamp_news_headline_random_test  lamp_scholarly_title_random_test lamp_tweet_random_test longlamp_abstract_generation_random_test longlamp_product_review_random_test longlamp_topic_writing_random_test 