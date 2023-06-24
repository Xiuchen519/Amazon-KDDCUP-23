
python ./RecStudio/run.py --model=GRU4Rec_Next_Feat --dataset=kdd_cup_2023_DE --gpu=0 \
 --layer_num=2 --dropout_rate=0.5 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/DE_data/DE_product_text_vectors.npy

python ./RecStudio/run.py --model=GRU4Rec_Next_Feat --dataset=kdd_cup_2023_JP --gpu=0 \
 --layer_num=2 --dropout_rate=0.5 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/JP_data/JP_product_text_vectors.npy

python ./RecStudio/run.py --model=GRU4Rec_Next_Feat --dataset=kdd_cup_2023_UK --gpu=0 \
 --layer_num=2 --dropout_rate=0.5 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/UK_data/UK_product_text_vectors.npy