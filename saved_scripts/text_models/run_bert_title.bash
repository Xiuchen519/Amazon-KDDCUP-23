CUDA_VISIBLE_DEVICES='4,5,6,7' WANDB_DISABLED='true' python -m torch.distributed.launch --nproc_per_node=4 \
 ./text_method/run_bert.py \
 --output_dir=./text_method/ckpt/06031745_title_UK_phase2/ \
 --tokenizer_name=roberta-base \
 --model_name_or_path=roberta-base \
 --dataset_name=kdd_cup_2023_UK \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_UK_bert.yaml \
 --do_train \
 --save_steps 5000 \
 --fp16 \
 --per_device_train_batch_size=42 \
 --neg_item_num=5 \
 --num_train_epochs=5 \
 --learning_rate 2e-5 \
 --dataloader_num_workers=8 \
  --use_session_title \
 --max_title_len=70 \
 --max_desc_len=70 \
 --negatives_x_device


CUDA_VISIBLE_DEVICES='4,5,6,7' WANDB_DISABLED='true' python -m torch.distributed.launch --nproc_per_node=4 \
 ./text_method/run_bert.py \
 --output_dir=./text_method/ckpt/06031745_title_DE_phase2/ \
 --tokenizer_name=bert-base-german-cased \
 --model_name_or_path=bert-base-german-cased \
 --dataset_name=kdd_cup_2023_DE \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_DE_bert.yaml \
 --do_train \
 --save_steps 5000 \
 --fp16 \
 --per_device_train_batch_size=42 \
 --neg_item_num=5 \
 --num_train_epochs=5 \
 --learning_rate 2e-5 \
 --dataloader_num_workers=8 \
 --use_session_title \
 --max_title_len=70 \
 --max_desc_len=70 \
 --negatives_x_device


CUDA_VISIBLE_DEVICES='4,5,6,7' WANDB_DISABLED='true' python -m torch.distributed.launch --nproc_per_node=4 \
 ./text_method/run_bert.py \
 --output_dir=./text_method/ckpt/06031745_title_JP_phase2/ \
 --tokenizer_name=cl-tohoku/bert-base-japanese-whole-word-masking \
 --model_name_or_path=cl-tohoku/bert-base-japanese-whole-word-masking \
 --dataset_name=kdd_cup_2023_JP \
 --dataset_config_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_JP_bert.yaml \
 --do_train \
 --save_steps 5000 \
 --fp16 \
 --per_device_train_batch_size=42 \
 --neg_item_num=5 \
 --num_train_epochs=5 \
 --learning_rate 2e-5 \
 --dataloader_num_workers=8 \
 --use_session_title \
 --max_title_len=70 \
 --max_desc_len=70 \
 --negatives_x_device
