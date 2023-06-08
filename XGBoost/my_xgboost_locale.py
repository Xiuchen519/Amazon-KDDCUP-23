# DE: 88913 JP: 78329 UK: 94574

import os
import numpy as np
import pandas as pd
import scipy.sparse as ssp
from functools import lru_cache
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import argparse
import random


import xgboost as xgb 
from sklearn.model_selection import GroupKFold
import time

parser = argparse.ArgumentParser()
parser.add_argument('--objective', type=str, default='rank:map')
parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--subsample', type=float, default=0.7)
parser.add_argument('--colsample', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--features', nargs='+', type=str, 
    default=['all_items_co_graph_count_0', 'all_items_co_graph_count_1', 'all_items_co_graph_count_2',
            'co_graph_counts_0',
            'co_graph_counts_1',
            'co_graph_counts_2',
            'desc_BM25_scores',
            'gru4rec_feat_scores_2',
            'gru4rec_scores_2',
            'narm_feat_scores',
            'narm_scores',
            'next_freq_',
            'normalized_all_items_co_graph_count_0',
            'normalized_all_items_co_graph_count_1',
            'normalized_all_items_co_graph_count_2',
            'normalized_co_graph_counts_0',
            'normalized_co_graph_counts_1',
            'normalized_co_graph_counts_2',
            'normalized_gru4rec_feat_scores_2',
            'normalized_gru4rec_scores_2',
            'normalized_narm_feat_scores',
            'normalized_narm_scores',
            'normalized_roberta_scores',
            'normalized_sasrec_cat_scores_3',
            'normalized_sasrec_duorec_score',
            'normalized_sasrec_feat_scores_3',
            'normalized_sasrec_scores_2',
            'normalized_sasrec_scores_3',
            'normalized_seqmlp_scores',
            'normalized_text_bert_scores',
            'normalized_title_bert_scores',
            'product_freq',
            'product_price',
            'roberta_scores',
            'sasrec_cat_scores_3',
            'sasrec_duorec_score',
            'sasrec_feat_scores_3',
            'sasrec_scores_2',
            'sasrec_scores_3',
            'seqmlp_scores',
            'sess_avg_price',
            'sess_locale',
            'text_bert_scores',
            'title_bert_scores',
            'title_BM25_scores',
            'w2v_l1_score',
            'w2v_l2_score',
            'w2v_l3_score'])

parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--early_stop_patience', type=int, default=500)
parser.add_argument('--merged_candidates_path', type=str, default='/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/XGBoost/candidates_phase2/merged_candidates_150_feature.parquet')
parser.add_argument('--gpu', type=int, default=3)
parser.add_argument('--sess_locale', type=str, default='DE')
args = parser.parse_args() 

sess_locale = args.sess_locale
# make dir
for x in ['all', 'DE', 'JP', 'UK']:

    if not os.path.exists(f'./XGBoost/logs/{x}'):
        os.makedirs(f'./XGBoost/logs/{x}')

    if not os.path.exists(f'./XGBoost/ckpt/{x}'):
        os.makedirs(f'./XGBoost/ckpt/{x}')



# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

merged_candidates_path = args.merged_candidates_path
def read_merged_candidates():
    return pd.read_parquet(merged_candidates_path, engine='pyarrow')


candidates_with_features = read_merged_candidates()
candidates_with_features['target'] = candidates_with_features['target'].astype(np.int32)
if args.sess_locale == 'all':
    candidates_with_features['sess_locale'] = candidates_with_features['sess_locale'].astype('category')
else:
    candidates_with_features = candidates_with_features.query(f'sess_locale=="{sess_locale}"').reset_index(drop=True)
    if 'sess_locale' in args.features:
        args.features.remove('sess_locale')


# FEATURES = set(candidates_with_features.columns)
# FEATURES.remove('sess_id'), FEATURES.remove('product'), FEATURES.remove('target')
# FEATURES.remove('sasrec_scores')
# FEATURES = list(FEATURES)
# FEATURES.sort()
FEATURES = args.features
FEATURES.sort()
FOLDS = 5
SEED = args.random_seed

random.seed(SEED)
# XGB MODEL PARAMETERS
xgb_parms = { 
    'max_depth': args.max_depth, 
    'learning_rate': args.learning_rate, 
    'subsample': args.subsample,
    'colsample_bytree': args.colsample, 
    'eval_metric': 'map@100-',
    'objective': args.objective,
    'scale_pos_weight': 200,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'random_state': SEED
}

skf = GroupKFold(n_splits=FOLDS)
cur_time = time.strftime(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime()))
log_path = f'./XGBoost/logs/{sess_locale}/XGB_{cur_time}.log'

with open(log_path, 'a') as f:
    f.write('Using Features: \n')
    f.write(f'{str(FEATURES)}\n')
    f.write('XGBoost parameters : \n')
    for k, v in xgb_parms.items():
        f.write(f'{k} : {v} \n')




for fold,(train_idx, valid_idx) in enumerate(skf.split(candidates_with_features, candidates_with_features['target'], groups=candidates_with_features['sess_id'] )):
    
    print('#'*25)
    print('### Fold',fold+1)
    print('### Train size',len(train_idx),'Valid size',len(valid_idx))
    print('#'*25)

    st_time = time.time()

    X_train = candidates_with_features.loc[train_idx, FEATURES]
    y_train = candidates_with_features.loc[train_idx, 'target']
    sess_id_train = candidates_with_features.loc[train_idx, ['sess_id', 'target']]
    group_size_train = sess_id_train.groupby(by='sess_id').count()['target'].to_numpy()

    X_valid = candidates_with_features.loc[valid_idx, FEATURES]
    y_valid = candidates_with_features.loc[valid_idx, 'target']

    sess_id_valid = candidates_with_features.loc[valid_idx, ['sess_id', 'target']]
    group_size_valid = sess_id_valid.groupby(by='sess_id').count()['target'].to_numpy()

    dtrain = xgb.DMatrix(X_train, y_train, group=group_size_train, enable_categorical=True) 
    dvalid = xgb.DMatrix(X_valid, y_valid, group=group_size_valid, enable_categorical=True) 

    res = {'train' : {'ndcg@100-' : []}, 'valid' : {'ndcg@100-' : []}}
    model = xgb.train(xgb_parms, 
        dtrain=dtrain,
        evals=[(dtrain,'train'),(dvalid,'valid')],
        num_boost_round=10000,
        early_stopping_rounds=args.early_stop_patience,
        evals_result=res,
        verbose_eval=100)
    
    ed_time = time.time()
    
    print(f'Running time : {(ed_time-st_time):.2f}s')

    with open(log_path, 'a') as f:
        f.write(f'Fold {fold+1}\n')
        f.write(f'Train size {len(train_idx)} Valid size {len(valid_idx)}\n')
        f.write(f'Running time {(ed_time-st_time):.2f}s\n')
        f.write(f'Best score : {model.best_score} Best iteration : {model.best_iteration}\n')

    model.save_model(f'./XGBoost/ckpt/{sess_locale}/XGB_{cur_time}_fold{fold}.json')