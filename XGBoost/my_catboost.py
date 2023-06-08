import os
import numpy as np
import pandas as pd
import scipy.sparse as ssp
from functools import lru_cache
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import argparse
import random

import catboost as cat 
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold
import time

parser = argparse.ArgumentParser()
parser.add_argument('--loss_function', type=str, default='PairLogitPairwise')
parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--subsample', type=float, default=0.7)
parser.add_argument('--colsample', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--features', nargs='+', type=str, 
    default=['product_freq', 'product_price',
             'sasrec_scores_2', 'normalized_sasrec_scores_2', 
             'sasrec_scores_3', 'normalized_sasrec_scores_3', 
             'gru4rec_scores', 'normalized_gru4rec_scores',
             'gru4rec_scores_2', 'normalized_gru4rec_scores_2',
             'roberta_scores', 'normalized_roberta_scores',
             'bert_scores', 'normalized_bert_scores',
             'co_graph_counts_0', 'normalized_co_graph_counts_0',
             'co_graph_counts_1', 'normalized_co_graph_counts_1',
             'co_graph_counts_2', 'normalized_co_graph_counts_2',
             'title_BM25_scores', 'desc_BM25_scores',
             'all_items_co_graph_count_0', 'normalized_all_items_co_graph_count_0',
             'all_items_co_graph_count_1', 'normalized_all_items_co_graph_count_1',
             'all_items_co_graph_count_2', 'normalized_all_items_co_graph_count_2',
             'seqmlp_scores', 'normalized_seqmlp_scores',
             'narm_scores', 'normalized_narm_scores',
             'sasrec_feat_scores', 'normalized_sasrec_feat_scores',
             'sasrec_text_scores', 'normalized_sasrec_text_scores',
             'sess_avg_price', 'sess_locale'])

parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--early_stop_patience', type=int, default=500)
parser.add_argument('--merged_candidates_path', type=str, default='/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/XGBoost/candidates_phase2/merged_candidates_cut_250_feature.parquet')
parser.add_argument('--gpu', type=int, default=1)
args = parser.parse_args() 

# make dir
if not os.path.exists('./XGBoost/cat_logs/'):
    os.makedirs('./XGBoost/cat_logs/')

if not os.path.exists('./XGBoost/cat_ckpt/'):
    os.makedirs('./XGBoost/cat_ckpt/')

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

merged_candidates_path = args.merged_candidates_path
def read_merged_candidates():
    return pd.read_parquet(merged_candidates_path, engine='pyarrow')


candidates_with_features = read_merged_candidates()
candidates_with_features['target'] = candidates_with_features['target'].astype(np.int32)


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
# CAT MODEL PARAMETERS
cat_parms = { 
    'max_depth': args.max_depth, 
    'learning_rate': args.learning_rate, 
    'subsample': args.subsample,
    # 'colsample_bytree': args.colsample, 
    'eval_metric': 'MRR:top=100',
    'loss_function': args.loss_function,
    'random_state': SEED
}

skf = GroupKFold(n_splits=FOLDS)
cur_time = time.strftime(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime()))

with open(f'./XGBoost/cat_logs/CAT_{cur_time}.log', 'a') as f:
    f.write('Using Features: \n')
    f.write(f'{str(FEATURES)}\n')
    f.write('CatBoost parameters : \n')
    for k, v in cat_parms.items():
        f.write(f'{k} : {v} \n')

for fold,(train_idx, valid_idx) in enumerate(skf.split(candidates_with_features, candidates_with_features['target'], groups=candidates_with_features['sess_id'] )):
    
    print('#'*25)
    print('### Fold',fold+1)
    print('### Train size', len(train_idx), 'Valid size', len(valid_idx))
    print('#'*25)

    st_time = time.time()

    model = CatBoostRanker(cat_features=['sess_locale'],
                           iterations=10000,
                           random_state=SEED,
                        #    subsample=args.subsample,
                           max_depth=args.max_depth,
                           learning_rate=args.learning_rate,
                           task_type='GPU',
                           loss_function=args.loss_function, 
                           eval_metric='MRR:top=100')
    
    train = Pool(
        cat_features=['sess_locale'],
        data = candidates_with_features.loc[train_idx, FEATURES],
        label = candidates_with_features.loc[train_idx, 'target'],
        group_id = candidates_with_features.loc[train_idx, 'sess_id']
    )
    
    valid = Pool(
        cat_features=['sess_locale'],
        data = candidates_with_features.loc[valid_idx, FEATURES],
        label = candidates_with_features.loc[valid_idx, 'target'],
        group_id = candidates_with_features.loc[valid_idx, 'sess_id']
    )
    
    model.fit(train,
              use_best_model=True,
              verbose=100,
              early_stopping_rounds=args.early_stop_patience,
              eval_set=valid,
    )

    ed_time = time.time()

    print(f'Running time : {(ed_time-st_time):.2f}s')

    with open(f'./XGBoost/cat_logs/CAT_{cur_time}.log', 'a') as f:
        f.write(f'Fold {fold+1}\n')
        f.write(f'Train size {len(train_idx)} Valid size {len(valid_idx)}\n')
        f.write(f'Running time {(ed_time-st_time):.2f}s\n')
        f.write(f'Best score : \n')
        for k in model.best_score_['validation']:
            f.write(f"{k} : {model.best_score_['validation'][k]} \n")
        f.write(f'Best iteration : {model.best_iteration_} \n')
        
    model.save_model(f'./XGBoost/cat_ckpt/CAT_{cur_time}_fold{fold}.cat')

