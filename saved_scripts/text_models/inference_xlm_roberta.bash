# phase 2 task1 valid query 
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --tokenizer_name=xlm-roberta-base \
 --output_dir=xlm-roberta_kdd_cup_finetune \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/06010126_xlm_roberta_phase2 \
 --dataset_name=kdd_cup_2023_roberta \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_roberta.yaml \
 --fp16 \
 --do_predict \
 --prediction_save_path=./text_method/phase2_task1_xlm_roberta_results/valid_results/ \
 --per_device_eval_batch_size=384 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=valid \
 --use_session_text \
 --max_title_len=50 \
 --max_desc_len=50 \


# phase 2 task1 test query 
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --tokenizer_name=xlm-roberta-base \
 --output_dir=xlm-roberta_kdd_cup_finetune \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/06010126_xlm_roberta_phase2 \
 --dataset_name=kdd_cup_2023_roberta \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_roberta.yaml \
 --fp16 \
 --do_predict \
 --prediction_data_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/task1_data/test_inter_feat_task1.csv \
 --prediction_save_path=./text_method/phase2_task1_xlm_roberta_results/test_results/ \
 --per_device_eval_batch_size=384 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=test \
 --use_session_text \
 --max_title_len=50 \
 --max_desc_len=50 \


# phase 2 task1 item query 
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --tokenizer_name=xlm-roberta-base \
 --output_dir=xlm-roberta_kdd_cup_finetune \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/06010126_xlm_roberta_phase2 \
 --dataset_name=kdd_cup_2023_roberta \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_roberta.yaml \
 --fp16 \
 --do_predict \
 --prediction_save_path=./text_method/phase2_task1_xlm_roberta_results/results/ \
 --per_device_eval_batch_size=512 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=all_items \
 --use_session_text \
 --max_title_len=50 \
 --max_desc_len=50 \
