

# phase 2 task1 valid query 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=roberta-base \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/05302029_UK_roberta_phase2 \
 --dataset_name=kdd_cup_2023_UK \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_UK_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_save_path=./text_method/phase2_task1_roberta_results_UK/valid_results/ \
 --per_device_eval_batch_size=256 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=valid \
 --use_session_text=True \


# phase 2 task1 test query 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=roberta-base \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/05302029_UK_roberta_phase2 \
 --dataset_name=kdd_cup_2023_UK \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_UK_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_data_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/task1_data/test_inter_feat_task1_UK.csv \
 --prediction_save_path=./text_method/phase2_task1_roberta_results_UK/test_results/ \
 --per_device_eval_batch_size=256 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=test \
 --use_session_text=True \


# phase 2 task1 valid query 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=roberta-base \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/05302029_UK_roberta_phase2 \
 --dataset_name=kdd_cup_2023_UK \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_UK_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_save_path=./text_method/phase2_task1_roberta_results_UK/results/ \
 --per_device_eval_batch_size=512 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=all_items \
 --use_session_text=True \


# phase 2 task1 valid query
CUDA_VISIBLE_DEVICES=2,0 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=bert-base-german-cased \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/05302029_DE_bert_phase2 \
 --dataset_name=kdd_cup_2023_DE \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_DE_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_save_path=./text_method/phase2_task1_bert_results_DE/valid_results/ \
 --per_device_eval_batch_size=384 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=valid \
 --use_session_title=True \

# phase 2 task1 test query
CUDA_VISIBLE_DEVICES=2,0 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=bert-base-german-cased \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/05302029_DE_bert_phase2 \
 --dataset_name=kdd_cup_2023_DE \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_DE_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_data_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/task1_data/test_inter_feat_task1_DE.csv \
 --prediction_save_path=./text_method/phase2_task1_bert_results_DE/test_results/ \
 --per_device_eval_batch_size=384 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=test \
 --use_session_title=True \


# phase 2 task1 test query
CUDA_VISIBLE_DEVICES=2,0 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=bert-base-german-cased \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/05302029_DE_bert_phase2 \
 --dataset_name=kdd_cup_2023_DE \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_DE_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_save_path=./text_method/phase2_task1_bert_results_DE/results/ \
 --per_device_eval_batch_size=384 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=all_items \
 --use_session_title=True \


# phase 2 task1 valid query
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=cl-tohoku/bert-base-japanese-whole-word-masking \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/06012057_JP_bert_phase2 \
 --dataset_name=kdd_cup_2023_JP \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_JP_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_save_path=./text_method/phase2_task1_bert_results_JP/valid_results/ \
 --per_device_eval_batch_size=384 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=valid \
 --use_session_title=True \

# phase 2 task1 test query
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=cl-tohoku/bert-base-japanese-whole-word-masking \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/06012057_JP_bert_phase2 \
 --dataset_name=kdd_cup_2023_JP \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_JP_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_data_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/task1_data/test_inter_feat_task1_JP_phase2.csv \
 --prediction_save_path=./text_method/phase2_task1_bert_results_JP/test_results/ \
 --per_device_eval_batch_size=384 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=test \
 --use_session_title=True \


# phase 2 task1 test query
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 \
 ./text_method/run_bert.py \
 --output_dir=bert_base_kdd_cup_finetune \
 --tokenizer_name=cl-tohoku/bert-base-japanese-whole-word-masking \
 --model_name_or_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/06012057_JP_bert_phase2 \
 --dataset_name=kdd_cup_2023_JP \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_JP_bert.yaml \
 --fp16 \
 --do_predict \
 --prediction_save_path=./text_method/phase2_task1_bert_results_JP/results/ \
 --per_device_eval_batch_size=384 \
 --dataloader_num_workers=8 \
 --eval_accumulation_steps=100 \
 --prediction_on=all_items \
 --use_session_title=True \
