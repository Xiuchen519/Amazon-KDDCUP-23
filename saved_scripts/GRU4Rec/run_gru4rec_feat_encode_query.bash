python ./RecStudio/run.py --model=GRU4Rec_Next_Feat --dataset=kdd_cup_2023_DE --do_encode_query --test_task=task1 \
 --layer_num=2 --dropout_rate=0.5 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 --batch_size=512 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/DE_data/DE_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/GRU4Rec_Feat/2023-06-03-03-05-40.ckpt --gpu=0

python ./RecStudio/run.py --model=GRU4Rec_Next_Feat --dataset=kdd_cup_2023_JP --do_encode_query --test_task=task1 \
 --layer_num=2 --dropout_rate=0.5 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 --batch_size=512 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/JP_data/JP_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/GRU4Rec_Feat/2023-06-03-03-05-43.ckpt --gpu=0

python ./RecStudio/run.py --model=GRU4Rec_Next_Feat --dataset=kdd_cup_2023_UK --do_encode_query --test_task=task1 \
 --layer_num=2 --dropout_rate=0.5 --use_product_feature=1 --use_text_feat=1 --feat_id_concat=0 --batch_size=512 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/UK_data/UK_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/GRU4Rec_Feat/2023-06-03-03-05-46.ckpt --gpu=0