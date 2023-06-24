# DE
python ./RecStudio/run.py --model=NARM_Feat --dataset=kdd_cup_2023_DE --do_encode_query \
 --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/DE_data/DE_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/saved/NARM_Feat/kdd_cup_2023_DE/2023-06-02-23-52-08.ckpt \
 --init_method=default --predict_topk=100 --gpu=0


# JP
python ./RecStudio/run.py --model=NARM_Feat --dataset=kdd_cup_2023_JP --do_encode_query \
 --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/JP_data/JP_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/saved/NARM_Feat/kdd_cup_2023_JP/2023-06-03-06-35-48.ckpt \
 --init_method=default --predict_topk=100 --gpu=0


# UK
python ./RecStudio/run.py --model=NARM_Feat --dataset=kdd_cup_2023_UK --do_encode_query \
 --layer_num=1 --dropout 0.25 0.5 --loss_func=softmax \
 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/UK_data/UK_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/saved/NARM_Feat/kdd_cup_2023_UK/2023-06-03-11-28-29.ckpt \
 --init_method=default --predict_topk=100 --gpu=0