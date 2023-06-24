
python ./RecStudio/run.py --model=NARM --dataset=kdd_cup_2023_DE --do_encode_query --test_task=task1 \
 --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/NARM/2023-05-30-01-52-10.ckpt --gpu=0


python ./RecStudio/run.py --model=NARM --dataset=kdd_cup_2023_JP --do_encode_query --test_task=task1 \
 --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/NARM/2023-05-30-19-13-45.ckpt --gpu=0


python ./RecStudio/run.py --model=NARM --dataset=kdd_cup_2023_UK --do_encode_query --test_task=task1 \
 --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/NARM/2023-05-30-11-02-11.ckpt --gpu=0