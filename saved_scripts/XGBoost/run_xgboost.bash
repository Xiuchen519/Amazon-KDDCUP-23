
python ./XGBoost/my_xgboost.py --objective rank:map \
 --features product_freq sess_locale \
 product_price sess_avg_price \
 sasrec_scores_2 normalized_sasrec_scores_2 \
 sasrec_scores_3 normalized_sasrec_scores_3 \
 gru4rec_scores_2 normalized_gru4rec_scores_2 \
 sasrec_feat_scores_3 normalized_sasrec_feat_scores_3 \
 sasrec_cat_scores_3 normalized_sasrec_cat_scores_3 \
 gru4rec_feat_scores_2 normalized_gru4rec_feat_scores_2 \
 narm_feat_scores normalized_narm_feat_scores \
 co_graph_counts_0 normalized_co_graph_counts_0 \
 co_graph_counts_1 normalized_co_graph_counts_1 \
 co_graph_counts_2 normalized_co_graph_counts_2 \
 roberta_scores normalized_roberta_scores \
 text_bert_scores normalized_text_bert_scores \
 title_bert_scores normalized_title_bert_scores \
 title_BM25_scores desc_BM25_scores feat_BM25_scores \
 min_max_title_BM25_scores min_max_desc_BM25_scores min_max_feat_BM25_scores \
 all_items_co_graph_count_0 normalized_all_items_co_graph_count_0 \
 all_items_co_graph_count_1 normalized_all_items_co_graph_count_1 \
 all_items_co_graph_count_2 normalized_all_items_co_graph_count_2 \
 seqmlp_scores normalized_seqmlp_scores \
 narm_scores normalized_narm_scores \
 sasrec_duorec_score normalized_sasrec_duorec_score \
 w2v_l1_score w2v_l2_score w2v_l3_score \
 next_freq_ \
 lyx_itemcf_score lyx_usercf_score \
 lyx_i2i_base_l1_score lyx_i2i_base_l2_score lyx_i2i_base_l3_score \
 normalized_lyx_i2i_base_l1_score normalized_lyx_i2i_base_l2_score normalized_lyx_i2i_base_l3_score \
 lyx_u2i_mbart_mean_score lyx_lknn_i2i_score lyx_lknn_u2i_score lyx_gru4rec_i2i_score lyx_gru4rec_u2i_score \
 normalized_lyx_u2i_mbart_mean_score normalized_lyx_lknn_i2i_score normalized_lyx_lknn_u2i_score normalized_lyx_gru4rec_i2i_score normalized_lyx_gru4rec_u2i_score \
 lyx_sasrec_u2i_nextitem_score lyx_sasrec_u2i_score_len12 lyx_sasrec_u2i_score_len13 \
 normalized_lyx_sasrec_u2i_nextitem_score normalized_lyx_sasrec_u2i_score_len12 normalized_lyx_sasrec_u2i_score_len13 \
 lyx_avghist_u2i_score lyx_avghist_i2i_score \
 lyx_w2v_cos_l1_score lyx_w2v_cos_l2_score lyx_w2v_cos_l3_score \
 --max_depth=4 \
 --early_stop_patience=500 \
 --gpu=2 \
 --merged_candidates_path=/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/XGBoost/candidates_phase2/merged_candidates_150_feature.parquet

