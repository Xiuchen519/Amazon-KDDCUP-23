import sys
sys.path = ['/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio'] + sys.path
import os
import logging

import faiss
import numpy as np
import pandas as pd 
from tqdm import tqdm
import pickle

from recstudio.data.advance_dataset import KDDCUPSessionDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class BaseFaissIPRetriever:
    def __init__(self, reps_dim: int):
        index = faiss.IndexFlatIP(reps_dim)
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), total=((num_query - 1) // batch_size + 1)):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0) # [N, topk]

        return all_scores, all_indices


def search_queries(retriever, q_reps, item_pos2id, item_id2token, depth, batch_size):
    if batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, depth, batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, depth)

    all_indices = all_indices + 1 # zero is for padding.
    item_indices = []
    for q_dd in tqdm(all_indices, total=len(all_indices)):
        tmp_indices = [] 
        for x in q_dd:
            tmp_indices.append(str(item_id2token[item_pos2id[x]]))
        item_indices.append(tmp_indices)
    item_indices = np.array(item_indices)
    return all_scores, item_indices


def write_ranking(item_indices, all_scores, ranking_save_file):
    ranking_df = pd.DataFrame({'next_item_prediction' : item_indices.tolist(), 'scores' : all_scores.tolist()})
    ranking_df.to_parquet(ranking_save_file, engine='pyarrow')
    logger.info(f'The ranking result is saved in {ranking_save_file}.')


def _load_cache(path):
    with open(path, 'rb') as f:
        download_obj = pickle.load(f)
    return download_obj


def load_datasets_from_cache(data_dir):
    logger.info(f"Load dataset from cache {data_dir}.")
    cache_datasets = _load_cache(data_dir)
    datasets = []
    for i in range(len(cache_datasets)):
        datasets.append(KDDCUPSessionDataset(None, data_dir, None, True))
        for k in cache_datasets[i].__dict__:
            attr = getattr(cache_datasets[i], k)
            setattr(datasets[i], k, attr)
    return datasets 


def search_by_faiss(query_reps_path, item_reps_path, dataset_cache_path, save_file, batch_size=512, depth=150, use_gpu=False):
    p_reps = np.load(os.path.join(item_reps_path, 'item.npy'))
    p_reps = np.array(p_reps).astype('float32')
    dataset = load_datasets_from_cache(dataset_cache_path)[0]
    item_pos2id = dataset.title_feat['product_id']
    item_id2token = dataset.field2tokens['product_id']
    print("shape of item", np.shape(p_reps))

    retriever = BaseFaissIPRetriever(np.shape(p_reps)[-1])

    faiss.omp_set_num_threads(64)
    if use_gpu:
        print('use GPU for Faiss')
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        retriever.index = faiss.index_cpu_to_all_gpus(
            retriever.index,
            co=co
        )

    retriever.add(p_reps)

    q_reps = np.load(os.path.join(query_reps_path, 'query.npy'))
    q_reps = np.array(q_reps).astype('float32')
    print("shape of query", np.shape(q_reps))

    logger.info('Index Search Start')
    all_scores, item_indices = search_queries(retriever, q_reps, item_pos2id, item_id2token, depth, batch_size)
    logger.info('Index Search Finished')

    write_ranking(item_indices, all_scores, save_file)
