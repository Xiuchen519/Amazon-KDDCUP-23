python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_DE --loss_func=softmax \
 --topk=300 --predict_topk=100 --batch_size=1024 --layer_num=2 --reverse_pos=1 --head_num=2 --embed_dim=128 \
 --scheduler=onplateau --scheduler_patience=2 --early_stop_patience=10 \
 --learning_rate=0.00075 --epochs=300 --gpu=0

python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_JP --loss_func=softmax \
 --topk=300 --predict_topk=100 --batch_size=1024 --layer_num=2 --reverse_pos=1 --head_num=2 --embed_dim=128 \
 --scheduler=onplateau --scheduler_patience=2 --early_stop_patience=10 \
 --learning_rate=0.00075 --epochs=300 --gpu=0

python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_UK --loss_func=softmax \
 --topk=300 --predict_topk=100 --batch_size=1024 --layer_num=2 --reverse_pos=1 --head_num=2 --embed_dim=128 \
 --scheduler=onplateau --scheduler_patience=2 --early_stop_patience=10 \
 --learning_rate=0.00075 --epochs=300 --gpu=0