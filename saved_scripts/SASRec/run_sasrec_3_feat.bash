
# cat
python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_DE --predict_topk=100 \
 --layer_num=3 --head_num=2 --use_color=0 --use_price=0 --use_text_feat=0 --use_product_feature=1 --embed_dim=128 --hidden_size=256 --init_method=default --scheduler=onplateau --gpu=0

# cat
python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_JP --predict_topk=100 \
 --layer_num=3 --head_num=2 --use_color=0 --use_price=0 --use_text_feat=0 --use_product_feature=1 --embed_dim=128 --hidden_size=256 --init_method=default --scheduler=onplateau --gpu=0

# cat
python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_UK --predict_topk=100 \
 --layer_num=3 --head_num=2 --use_color=0 --use_price=0 --use_text_feat=0 --use_product_feature=1 --embed_dim=128 --hidden_size=256 --init_method=default --scheduler=onplateau --gpu=0


# cat + text
python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_DE --layer_num=3 --reverse_pos=1 --head_num=2 --embed_dim=128 \
 --use_text_feat=1 --use_product_feature=1 --use_color=0 --use_price=0 \
 --feat_id_concat=0 --item_text_layers 768 128 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/JP_data/DE_product_text_vectors.npy --predict_topk=100 --gpu=0


# cat + text
python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_JP --layer_num=3 --reverse_pos=1 --head_num=2 --embed_dim=128 \
 --use_text_feat=1 --use_product_feature=1 --use_color=0 --use_price=0 \
 --feat_id_concat=0 --item_text_layers 768 128 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/JP_data/JP_product_text_vectors.npy --predict_topk=100 --gpu=0


# cat + text
python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_UK --layer_num=3 --reverse_pos=1 --head_num=2 --embed_dim=128 \
 --use_text_feat=1 --use_product_feature=1 --use_color=0 --use_price=0 \
 --feat_id_concat=0 --item_text_layers 768 128 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/JP_data/UK_product_text_vectors.npy --predict_topk=100 --gpu=0