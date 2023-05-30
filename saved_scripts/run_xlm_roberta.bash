CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 \
 ./text_method/run_bert.py \
 --output_dir=./text_method/ckpt/xlm_roberta/ \
 --tokenizer_name=xlm-roberta-base \
 --model_name_or_path=xlm-roberta-base \
 --dataset_name=kdd_cup_2023_roberta \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_roberta.yaml \
 --do_train \
 --save_steps 5000 \
 --fp16 \
 --per_device_train_batch_size=45 \
 --neg_item_num=6 \
 --num_train_epochs=4 \
 --learning_rate 2e-5 \
 --dataloader_num_workers=8 \
 --negatives_x_device \
 --use_session_text \
 --max_title_len=50 \
 --max_desc_len=50 \