# Phase2 task1 test
CUDA_VISIBLE_DEVICES=1,3 python ./text_method/test.py \
 --query_reps_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_xlm_roberta_results/test_results/test_query_reps \
 --item_reps_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_xlm_roberta_results/results/item_reps \
 --dataset_cache_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/.recstudio/cache/425c48dedf55cb3d7582fa2c7b6216dc \
 --ranking_file /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/phase2_task1_xlm_roberta_results/test_results/phase2_task1_test_300_with_score.parquet \
 --depth=350 \
 --use_gpu 