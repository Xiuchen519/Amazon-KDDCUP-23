import numpy as np
import pandas as pd
import scipy.sparse as ssp
from functools import lru_cache
from tqdm import tqdm, trange
from collections import Counter, defaultdict

import xgboost as xgb 
from sklearn.model_selection import GroupKFold
import time

merged_candidates_path = '/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/XGBoost/candidates/merged_candidates_feature_2.parquet'
def read_merged_candidates():
    return pd.read_parquet(merged_candidates_path, engine='pyarrow')


candidates_with_features = read_merged_candidates()
candidates_with_features['target'] = candidates_with_features['target'].astype(np.int32)


FEATURES = set(candidates_with_features.columns)
FEATURES.remove('sess_id'), FEATURES.remove('product'), FEATURES.remove('sess_locale'), FEATURES.remove('target')
FEATURES = list(FEATURES)
FEATURES.sort()
FOLDS = 5
SEED = 42
LR = 0.1

# XGB MODEL PARAMETERS
xgb_parms = { 
    'max_depth': 4, 
    'learning_rate': LR, 
    'subsample': 0.7,
    'colsample_bytree': 0.5, 
    'eval_metric': 'ndcg@100-',
    'objective': 'binary:logistic',
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

    dtrain = xgb.DMatrix(X_train, y_train, group=group_size_train) 
    dvalid = xgb.DMatrix(X_valid, y_valid, group=group_size_valid) 

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

    model.save_model(f'./XGBoost/ckpt/XGB_{cur_time}_fold{fold}.xgb')