CUDA_VISIBLE_DEVICES='1,3,4,5' WANDB_DISABLED='true' python -m torch.distributed.launch --nproc_per_node=4 \
 ./text_method/run_bert.py \
 --output_dir=./text_method/ckpt/05302029_UK_roberta_phase2/ \
 --tokenizer_name=roberta-base \
 --model_name_or_path=roberta-base \
 --dataset_name=kdd_cup_2023_UK \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_UK_bert.yaml \
 --do_train \
 --save_steps 4000 \
 --fp16 \
 --per_device_train_batch_size=28 \
 --neg_item_num=4 \
 --num_train_epochs=5 \
 --learning_rate 2e-5 \
 --dataloader_num_workers=8 \
 --use_session_text \
 --max_title_len=50 \
 --max_desc_len=50 \
 --negatives_x_device


CUDA_VISIBLE_DEVICES='1,3,4,5' WANDB_DISABLED='true' python -m torch.distributed.launch --nproc_per_node=4 \
 ./text_method/run_bert.py \
 --output_dir=./text_method/ckpt/05302029_DE_bert_phase2/ \
 --tokenizer_name=bert-base-german-cased \
 --model_name_or_path=bert-base-german-cased \
 --dataset_name=kdd_cup_2023_DE \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_DE_bert.yaml \
 --do_train \
 --save_steps 4000 \
 --fp16 \
 --per_device_train_batch_size=28 \
 --neg_item_num=4 \
 --num_train_epochs=5 \
 --learning_rate 2e-5 \
 --dataloader_num_workers=8 \
 --use_session_text \
 --max_title_len=50 \
 --max_desc_len=50 \
 --negatives_x_device


CUDA_VISIBLE_DEVICES='0,3,6,7' WANDB_DISABLED='true' python -m torch.distributed.launch --nproc_per_node=4 \
 ./text_method/run_bert.py \
 --output_dir=./text_method/ckpt/06012057_JP_bert_phase2/ \
 --tokenizer_name=cl-tohoku/bert-base-japanese-whole-word-masking \
 --model_name_or_path=cl-tohoku/bert-base-japanese-whole-word-masking \
 --dataset_name=kdd_cup_2023_JP \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_JP_bert.yaml \
 --do_train \
 --save_steps 4000 \
 --fp16 \
 --per_device_train_batch_size=28 \
 --neg_item_num=4 \
 --num_train_epochs=5 \
 --learning_rate 2e-5 \
 --dataloader_num_workers=8 \
 --use_session_text \
 --max_title_len=50 \
 --max_desc_len=50 \
 --negatives_x_device