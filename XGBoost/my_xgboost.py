import os
import numpy as np
import pandas as pd
import scipy.sparse as ssp
from functools import lru_cache
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import argparse


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
    default=['product_freq', 'product_price',
             'sasrec_scores_2', 'sasrec_normalized_scores_2', 
             'sasrec_scores_3', 'sasrec_normalized_scores_3', 
             'gru4rec_scores', 'gru4rec_normalized_scores',
             'gru4rec_scores_2', 'gru4rec_normalized_scores_2',
             'roberta_scores', 'roberta_normalized_scores',
             'co_graph_counts_0', 'normalized_co_graph_counts_0',
             'co_graph_counts_1', 'normalized_co_graph_counts_1',
             'co_graph_counts_2', 'normalized_co_graph_counts_2',
             'title_BM25_scores', 'desc_BM25_scores',
             'all_items_co_graph_count_0', 'normalized_all_items_co_graph_count_0',
             'seqmlp_scores', 'seqmlp_normalized_scores',
             'sess_avg_price', 'sess_locale'])
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--merged_candidates_path', type=str, default='/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/XGBoost/candidates/merged_candidates_no_hist_feature.parquet')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args() 

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

merged_candidates_path = args.merged_candidates_path
def read_merged_candidates():
    return pd.read_parquet(merged_candidates_path, engine='pyarrow')


candidates_with_features = read_merged_candidates()
candidates_with_features['target'] = candidates_with_features['target'].astype(np.int32)
candidates_with_features['sess_locale'] = candidates_with_features['sess_locale'].astype('category')


# FEATURES = set(candidates_with_features.columns)
# FEATURES.remove('sess_id'), FEATURES.remove('product'), FEATURES.remove('target')
# FEATURES.remove('sasrec_scores')
# FEATURES = list(FEATURES)
# FEATURES.sort()
FEATURES = args.features
FEATURES.sort()
FOLDS = 5
SEED = args.random_seed

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

with open(f'./XGBoost/logs/XGB_{cur_time}.log', 'a') as f:
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
        early_stopping_rounds=200,
        evals_result=res,
        verbose_eval=100)
    
    ed_time = time.time()
    
    print(f'Running time : {(ed_time-st_time):.2f}s')

    with open(f'./XGBoost/logs/XGB_{cur_time}.log', 'a') as f:
        f.write(f'Fold {fold+1}\n')
        f.write(f'Train size {len(train_idx)} Valid size {len(valid_idx)}\n')
        f.write(f'Running time {(ed_time-st_time):.2f}s\n')
        f.write(f'Best score : {model.best_score} Best iteration : {model.best_iteration}\n')

    model.save_model(f'./XGBoost/ckpt/XGB_{cur_time}_fold{fold}.json')