# Phase2 task1 test
CUDA_VISIBLE_DEVICES=0 python ./text_method/test.py \
 --query_reps_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_bert_title_results_DE/valid_results/valid_query_reps \
 --item_reps_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_bert_title_results_DE/results/item_reps \
 --dataset_cache_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/.recstudio/cache/87c62409540df6ccca9d90ab244af0e5 \
 --ranking_file /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_bert_title_results_DE/valid_results/phase2_task1_valid_300_with_score_DE.parquet \
 --depth=350 \
 --use_gpu 

# Phase2 task1 test
CUDA_VISIBLE_DEVICES=0 python ./text_method/test.py \
 --query_reps_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_bert_title_results_JP/valid_results/valid_query_reps \
 --item_reps_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_bert_title_results_JP/results/item_reps \
 --dataset_cache_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/.recstudio/cache/d296613e4d5aa97bebf6c4b114f02d89 \
 --ranking_file /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_bert_title_results_JP/valid_results/phase2_task1_valid_300_with_score_JP.parquet \
 --depth=350 \
 --use_gpu 

# Phase2 task1 test
CUDA_VISIBLE_DEVICES=0 python ./text_method/test.py \
 --query_reps_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_roberta_title_results_UK/valid_results/valid_query_reps \
 --item_reps_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_roberta_title_results_UK/results/item_reps \
 --dataset_cache_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/.recstudio/cache/d3540d1aadb28b19da92e77c7cf0f7e2 \
 --ranking_file /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_roberta_title_results_UK/valid_results/phase2_task1_valid_300_with_score_UK.parquet \
 --depth=350 \
 --use_gpu 