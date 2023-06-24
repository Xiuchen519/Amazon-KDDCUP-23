
# UK
python ./RecStudio/run.py --model=NARM --dataset=kdd_cup_2023_DE --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax --init_method=default --predict_topk=100 --gpu=0

# DE
python ./RecStudio/run.py --model=NARM --dataset=kdd_cup_2023_UK --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax --init_method=default --predict_topk=100 --gpu=0

# JP
python ./RecStudio/run.py --model=NARM --dataset=kdd_cup_2023_JP --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax --init_method=default --predict_topk=100 --gpu=0
