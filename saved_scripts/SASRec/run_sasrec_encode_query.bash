
python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_DE --do_encode_query --test_task=task1 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 --batch_size=512 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Next/three_layers/2023-05-30-01-31-31.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_JP --do_encode_query --test_task=task1 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 --batch_size=512 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Next/three_layers/2023-05-30-13-33-29.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_UK --do_encode_query --test_task=task1 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 --batch_size=512 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Next/three_layers/2023-05-30-00-56-53.ckpt --gpu=0


python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_DE --do_encode_query --test_task=task1 \
 --layer_num=2 --head_num=2 --hidden_size=256 --dropout=0.5 --batch_size=512 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Next/two_layers/2023-05-30-13-04-14.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_JP --do_encode_query --test_task=task1 \
 --layer_num=2 --head_num=2 --hidden_size=256 --dropout=0.5 --batch_size=512 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Next/two_layers/2023-05-30-23-06-00.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next --dataset=kdd_cup_2023_UK --do_encode_query --test_task=task1 \
 --layer_num=2 --head_num=2 --hidden_size=256 --dropout=0.5 --batch_size=512 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Next/two_layers/2023-05-30-02-01-23.ckpt --gpu=0


python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_DE --do_encode_query --test_task=task1 \
 --use_text_feat=0 --use_product_feature=1 --feat_id_concat=0 --use_color=0 --use_price=0 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Feat_Cat/2023-06-01-01-38-30.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_JP --do_encode_query --test_task=task1 \
 --use_text_feat=0 --use_product_feature=1 --feat_id_concat=0 --use_color=0 --use_price=0 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Feat_Cat/2023-06-01-11-34-52.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_UK --do_encode_query --test_task=task1 \
 --use_text_feat=0 --use_product_feature=1 --feat_id_concat=0 --use_color=0 --use_price=0 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Feat_Cat/2023-06-01-12-17-17.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_DE --do_encode_query --test_task=task1 \
 --use_text_feat=1 --use_product_feature=1 --feat_id_concat=0 --use_color=0 --use_price=0 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/DE_data/DE_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Feat/2023-06-01-18-30-22.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_JP --do_encode_query --test_task=task1 \
 --use_text_feat=1 --use_product_feature=1 --feat_id_concat=0 --use_color=0 --use_price=0 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/JP_data/JP_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Feat/2023-06-02-12-45-35.ckpt --gpu=0

python ./RecStudio/run.py --model=SASRec_Next_Feat --dataset=kdd_cup_2023_UK --do_encode_query --test_task=task1 \
 --use_text_feat=1 --use_product_feature=1 --feat_id_concat=0 --use_color=0 --use_price=0 \
 --layer_num=3 --head_num=2 --hidden_size=256 --dropout=0.5 \
 --item_text_vectors=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/UK_data/UK_product_text_vectors.npy \
 --model_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/best_ckpt_phase2/logs/SASRec_Feat/2023-06-02-14-01-25.ckpt --gpu=0

