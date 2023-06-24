# printf "*********** start to merge price ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_price.ipynb"

# endtime=`date +%s`

# echo "*********** price is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge item_freq ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_item_freq.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_item_freq.ipynb

# endtime=`date +%s`

# echo "*********** item_freq is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge sasrec_2 score ***********\n"

# starttime=`date +%s`

# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_sasrec_score.ipynb

# endtime=`date +%s`

# echo "*********** sasrec_2 score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge sasrec_3 score ***********\n"

# starttime=`date +%s`

# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_sasrec_3_score.ipynb

# endtime=`date +%s`

# echo "*********** sasrec_3 score is added ***********"
# echo "running time : "$((endtime - starttime))"s"



# printf "*********** start to merge seqmlp score ***********\n"

# starttime=`date +%s`

# # jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_seqmlp_score.ipynb
# ipython -c "%run ./XGBoost/merge_features/merge_seqmlp_score.ipynb"

# endtime=`date +%s`

# echo "*********** seqmlp score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge roberta score ***********\n"

# starttime=`date +%s`

# # jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_roberta_score.ipynb
# ipython -c "%run ./XGBoost/merge_features/test/merge_roberta_score.ipynb"

# endtime=`date +%s`

# echo "*********** roberta score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge narm score ***********\n"

# starttime=`date +%s`

# # jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_narm_score.ipynb
# ipython -c "%run ./XGBoost/merge_features/test/merge_narm_score.ipynb"

# endtime=`date +%s`

# echo "*********** narm score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge gru4rec score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_gru_score.ipynb"

# endtime=`date +%s`

# echo "*********** gru4rec score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge gru4rec_2 score ***********\n"

# starttime=`date +%s`

# # jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_gru_score_2.ipynb
# ipython -c "%run ./XGBoost/merge_features/test/merge_gru_score_2.ipynb"

# endtime=`date +%s`

# echo "*********** gru4rec_2 score is added ***********"
# echo "running time : "$((endtime - starttime))"s"



# printf "*********** start to merge title BM25 score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_title_bm25_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_title_bm25_score.ipynb

# endtime=`date +%s`

# echo "*********** title BM25 is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge desc BM25 score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_desc_bm25_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_desc_bm25_score.ipynb

# endtime=`date +%s`

# echo "*********** desc BM25 is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge all items co graph score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_all_items_co_graph_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_all_items_co_graph_score.ipynb

# endtime=`date +%s`

# echo "*********** all items co graph score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge co graph score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_co_graph_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_co_graph_score.ipynb


# endtime=`date +%s`

# echo "*********** co graph score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge bert score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_bert_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_bert_score.ipynb

# endtime=`date +%s`

# echo "*********** bert score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge xlm-roberta score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_roberta_score.ipynb"
# # jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_roberta_score.ipynb

# endtime=`date +%s`

# echo "*********** xlm-roberta score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge sasrec feat score 3 ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_sasrec_feat_3_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_sasrec_feat_3_score.ipynb


# endtime=`date +%s`

# echo "*********** sasrec feat score 3 is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge sasrec cat score 3 ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_sasrec_cat_3_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_sasrec_cat_3_score.ipynb


# endtime=`date +%s`

# echo "*********** sasrec cat score 3 is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge narm feat score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_narm_feat_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_narm_feat_score.ipynb


# endtime=`date +%s`

# echo "*********** narm feat score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge gru4rec feat score 2 ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_gru4rec_feat_2_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_gru4rec_feat_2_score.ipynb


# endtime=`date +%s`

# echo "*********** gru4rec feat score 2 is added ***********"
# echo "running time : "$((endtime - starttime))"s"


# printf "*********** start to merge title bert score ***********\n"

# starttime=`date +%s`

# ipython -c "%run ./XGBoost/merge_features/test/merge_bert_title_score.ipynb"
# jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/merge_bert_title_score.ipynb


# endtime=`date +%s`

# echo "*********** title bert score is added ***********"
# echo "running time : "$((endtime - starttime))"s"


printf "*********** start to merge gru4rec cat score ***********\n"

starttime=`date +%s`

ipython -c "%run ./XGBoost/merge_features/merge_gru4rec_cat_2_score.ipynb"
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/test/merge_gru4rec_cat_2_score.ipynb


endtime=`date +%s`

echo "*********** gru4rec cat score is added ***********"
echo "running time : "$((endtime - starttime))"s"



printf "*********** start to merge narm cat score ***********\n"

starttime=`date +%s`

ipython -c "%run ./XGBoost/merge_features/merge_narm_cat_score.ipynb"
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=pytorch_gpu --ExecutePreprocessor.timeout=-1 --inplace --execute ./XGBoost/merge_features/test/merge_narm_cat_score.ipynb


endtime=`date +%s`

echo "*********** narm cat score is added ***********"
echo "running time : "$((endtime - starttime))"s"

