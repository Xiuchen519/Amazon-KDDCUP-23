# CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 \
#  -m bi_encoder.run \
#  --output_dir retromae_kdd_cup_passage_fintune \
#  --model_name_or_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RetroMAE/examples/retriever/kdd_cup/ckpt/04051843  \
#  --corpus_file ./data/BertTokenizer_data/UK_corpus \
#  --passage_max_len 50 \
#  --fp16  \
#  --do_predict \
#  --prediction_save_path results/ \
#  --per_device_eval_batch_size 256 \
#  --dataloader_num_workers 6 \
#  --eval_accumulation_steps 100 

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 \
-m bi_encoder.run \
--output_dir retromae_msmarco_passage_fintune \
--model_name_or_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RetroMAE/examples/retriever/kdd_cup/ckpt/04051843 \
--test_query_file ./data/BertTokenizer_data/valid_UK_query \
--query_max_len 260 \
--fp16  \
--do_predict \
--prediction_save_path valid_results/ \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100 

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 \
-m bi_encoder.run \
--output_dir retromae_msmarco_passage_fintune \
--model_name_or_path /root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RetroMAE/examples/retriever/kdd_cup/ckpt/04051843 \
--test_query_file ./data/BertTokenizer_data/test1_UK_query \
--query_max_len 260 \
--fp16  \
--do_predict \
--prediction_save_path test_results/ \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100 