
# DE
python ./RecStudio/run.py --model=NARM_Feat --dataset=kdd_cup_2023_DE --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --use_product_feature=1 --use_text_feat=0 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/DE_data/DE_product_text_vectors.npy \
 --init_method=default --predict_topk=100 --gpu=0

# JP
python ./RecStudio/run.py --model=NARM_Feat --dataset=kdd_cup_2023_JP --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --use_product_feature=1 --use_text_feat=0 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/JP_data/JP_product_text_vectors.npy \
 --init_method=default --predict_topk=100 --gpu=0

# UK
python ./RecStudio/run.py --model=NARM_Feat --dataset=kdd_cup_2023_UK --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --use_product_feature=1 --use_text_feat=0 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/UK_data/UK_product_text_vectors.npy \
 --init_method=default --predict_topk=100 --gpu=0

