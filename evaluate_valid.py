## Evaluate on valid dataset for task1
## Input: the file of prediction file which consist two columns: ['next_item_prediction', 'locale']
## Output: the metrics, such as MRR@100, Hit@100, DCG@100

import os
import argparse

WORK_DIR = os.path.dirname(__file__)

import numpy as np
import pandas as pd

def get_rank(truth: str, candidates: list) -> int:
    try:
        rank = candidates.index(truth) + 1
    except ValueError:
        rank = -1
    return rank

def group_pointwise_df(df: pd.DataFrame) -> pd.DataFrame:
    res = df.groupby("sess_id").agg(list)
    res['next_item'] = res['product_id'].apply(lambda x: x[-1])
    res['product_id'] = res['product_id'].apply(lambda x: x[:-1])
    res['locale'] = res["locale"].apply(lambda x: x[0])
    res = res.rename(columns={'product_id': 'prev_items'})
    return res

def evaluate(tgt_df, pred_df) -> dict:
    assert len(tgt_df) == len(pred_df), 'Length not equal.'
    assert (tgt_df['locale'] == pred_df['locale']).sum() == len(tgt_df), 'Locale not consisent.'
    if 'next_item_prediction' not in pred_df:
        if 'candidates' in pred_df:
            pred_df = pred_df.rename(columns = {'candidates': 'next_item_prediction'})
    pred_list_len = pred_df['next_item_prediction'].apply(len)
    if pred_list_len.var() == 0:
        k = pred_list_len[0]
    else:
        k = 'ALL'
        avg_len = pred_list_len.mean()
        print("Mean length of prediction = {:.4f}".format(avg_len))
    truth = tgt_df['next_item'].tolist()
    pred = pred_df['next_item_prediction'].tolist()
    
    hit = 0
    mr = 0.0
    mrr = 0.0
    dcg = 0.0
    for i in range(len(truth)):
        rank = get_rank(truth[i], pred[i].tolist())
        if rank < 0:
            pass
        else:
            hit += 1
            mr += rank
            mrr += 1.0 / rank
            dcg += 1.0 / np.log2(rank + 1)
    return {f'dcg@{k}': dcg/len(truth), f'hit@{k}': hit/len(truth), f'mr@{k}': mr/len(truth), f'mrr@{k}': mrr/len(truth)}

def main():
    parser = argparse.ArgumentParser('Task1 Validation Evaluator')
    parser.add_argument('-f', '--file', type=str, help='prediction files based WORK_DIR')
    parser.add_argument('-v', '--valid_type', type=str, choices=['all', 'tune'], help='all data or tune data')
    args, _ = parser.parse_known_args()

    valid_file_path = os.path.join(WORK_DIR, f'data_for_recstudio/{args.valid_type}_task_1_valid_inter_feat.csv')
    valid_df = pd.read_csv(valid_file_path)
    valid_df = group_pointwise_df(valid_df)
    valid_df = valid_df[['locale', 'next_item']]
    
    if os.path.exists(args.file):
        pred_file_path = args.file
    else:
        pred_file_path = os.path.join(WORK_DIR, args.file)
    
    pred_df = pd.read_parquet(pred_file_path, engine='pyarrow')   # ['locale', 'next_item_prediction']

    result = evaluate(valid_df, pred_df)
    print(result)


if __name__ == '__main__':
    main()
