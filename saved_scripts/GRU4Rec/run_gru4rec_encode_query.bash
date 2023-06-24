
python ./RecStudio/run.py --model=GRU4Rec_Next --dataset=kdd_cup_2023_DE --do_encode_query --test_task=task1 \
 --batch_size=512 --layer_num=2 --dropout_rate=0.5 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/GRU4Rec_Next/two_layers/2023-05-30-14-43-10.ckpt --gpu=3

python ./RecStudio/run.py --model=GRU4Rec_Next --dataset=kdd_cup_2023_JP --do_encode_query --test_task=task1 \
 --batch_size=512 --layer_num=2 --dropout_rate=0.5 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/GRU4Rec_Next/two_layers/2023-05-30-20-23-17.ckpt --gpu=3

python ./RecStudio/run.py --model=GRU4Rec_Next --dataset=kdd_cup_2023_UK --do_encode_query --test_task=task1 \
 --batch_size=512 --layer_num=2 --dropout_rate=0.5 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/GRU4Rec_Next/two_layers/2023-05-31-00-43-18.ckpt --gpu=3