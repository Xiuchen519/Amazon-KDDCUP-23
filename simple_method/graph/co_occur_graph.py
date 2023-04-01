import dgl
import random
import pickle
import torch as th 
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import *
import scipy.sparse as ssp


def group_pointwise_df(df: pd.DataFrame) -> pd.DataFrame:
    res = df.groupby("sess_id").agg(list)
    res['next_item'] = res['product_id'].apply(lambda x: x[-1])
    res['product_id'] = res['product_id'].apply(lambda x: x[:-1])
    res['locale'] = res['locale'].apply(lambda x: x[0])
    res = res.rename(columns={'product_id': 'prev_items'})
    return res


def get_sessions(df: pd.DataFrame, id_dict: dict, use_last_item: bool=False, original_file=False, return_last_only: bool=False) -> list:
    if 'next_item' in df and use_last_item:
        if not original_file:
            all_item = df.apply(lambda x:x['prev_items'] + [x['next_item']], axis=1)
        else:
            all_item = df.apply(lambda x: eval((x['prev_items'][:-1]+f" '{x['next_item']}']").replace(" ", ",")), axis=1)
    else:
        if not original_file:
            all_item = df['prev_items']
        else:
            all_item = df.apply(lambda x: eval(x['prev_items'].replace(" ", ",")), axis=1)
    all_item_id = []
    for x in all_item:
        if return_last_only:
            all_item_id.append(id_dict[x[-1]])
        else:
            all_item_id.append([id_dict[y] for y in x])
    return all_item_id


def get_pop_items(df: pd.DataFrame, pop_k: int):
    item_counts = df['product_id'].value_counts(sort=True, ascending=False)
    pop_item = item_counts.keys()[:pop_k].tolist()
    return pop_item



class CoOccuGraph():
    def __init__(self, item_map: dict, pop_k: int=200, locale_sensitive: bool=True) -> None:
        # self.train_sess_item_list = get_sessions(group_pointwise_df(train_df), item_map, False, False)
        # self.test_sess_item_list = get_sessions(group_pointwise_df(test_df), item_map, False, False)
        self.item_map = item_map
        self.num_items = max(item_map.values()) + 1 # including padding idx
        self.directional = False
        self.graph = None
        self.pop_k = pop_k
        self.locale_sensitive = locale_sensitive
        self.pop_items = None
        self.id2pid = {v: k for k,v in item_map.items()}
        # self.graph = self.construct_graph(self.train_sess_item_list, directional, verbose=True)
        pass


    def construct_graph(self, train_df: pd.DataFrame, use_last_item: bool=True, original_file: bool=False, directional: bool=False, verbose: bool=True) -> ssp.csr_matrix:
        print("Building graph...")
        if not original_file:
            df = group_pointwise_df(train_df)
        else:
            df = train_df
        sess_id_list = get_sessions(df, self.item_map, test=~use_last_item, original_file=original_file)
        if verbose:
            iterator = tqdm(sess_id_list)
        else:
            iterator = sess_id_list

        res = {}
        for l in iterator:
            if len(l) <= 1:
                continue
            for i, item1 in enumerate(l):
                if id not in res:
                    res[item1] = Counter()
                for j in range(i+1, len(l)):
                    item2 = l[j]
                    res[item1][item2] += 1
                    if not directional:
                        if item2 not in res:
                            res[item2] = Counter()
                        res[item2][item1] += 1

        num_edges = sum([len(v) for k,v in res.items()])
        graph_edge = np.zeros((num_edges, 2), dtype=int)
        edge_weights = np.zeros((num_edges), dtype=int)
        i = 0
        for k, v in res.items():
            for k1, v1 in v.items():
                graph_edge[i][0] = k 
                graph_edge[i][1] = k1
                edge_weights[i] = v1
                i += 1
        assert i == num_edges
        
        adj_matrix = ssp.coo_matrix(
            (edge_weights, (graph_edge[:, 0], graph_edge[:, 1])), 
            shape=(self.num_items, self.num_items)
            )
        self.graph = adj_matrix.tocsr()
        self.directional = directional
        if self.locale_sensitive:
            self.pop_items = {
                locale: get_pop_items(train_df[train_df['locale']==locale], self.pop_k) 
                for locale in train_df['locale'].unique()}
        else:
            global_pop_items = get_pop_items(train_df, self.pop_k)
            self.pop_items = {
                locale: global_pop_items for locale in train_df['locale'].unique()}
        print("Graph is built sucessfully.")
        return self.graph


    def get_neighbors(self, item_id: int, sort: bool=True) -> tuple:
        indptr = self.graph.indptr[item_id: item_id+2]
        indices = self.graph.indices[indptr[0]:indptr[1]]
        data = self.graph.data[indptr[0]:indptr[1]]
        if sort:
            idx = np.argsort(- data)
            indices = indices[idx]
            data = data[idx]
        return indices, data


    def get_neighbors_for_list(self, item_list: list, sort: bool=True) -> tuple:
        indices = np.array([])
        data = np.array([])
        for item in item_list:
            neighbors, weight = self.get_neighbors(item, False)
            indices = np.concatenate((indices, neighbors))
            data = np.concatenate((data, weight))
        if sort:
            idx = np.argsort(- data)
            indices = indices[idx]
            data = data[idx]
        return indices, data


    def predict(self, test_df: pd.DataFrame, last_item_only: bool=True, k: int=None, original_file: bool=False, verbose: bool=True):
        if not original_file:
            test_df = group_pointwise_df(test_df)
        sess_id_list = get_sessions(test_df, self.item_map, True, original_file, last_item_only)
        sess_locale = test_df['locale'].tolist()
        if verbose:
            iterator = tqdm(enumerate(sess_id_list), total=len(sess_id_list))
        else:
            iterator = enumerate(sess_id_list)

        res = []
        for i, item in iterator:
            if last_item_only:
                neighbors, _ = self.get_neighbors(item)
            else:
                neighbors, _ = self.get_neighbors_for_list(item)
            if k is not None:   # pad/cut result to specific length
                if len(neighbors) <= k:
                    locale = sess_locale[i]
                    pop_items = random.sample(self.pop_items[locale], k-len(neighbors))
                    res.append([self.id2pid[x] for x in neighbors]+pop_items)
                else:
                    res.append([self.id2pid[x] for x in neighbors[:k]])
            else:
                res.append([self.id2pid[x] for x in neighbors])
        
        return pd.DataFrame({'next_item_prediction': res, 'locale': sess_locale})
    
    def save(self, fname: str):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        print(f'Graph data saved in {fname}.')

    def add(self, graph: ssp.csr_matrix):
        assert self.graph.shape == graph.shape, 'The shape of graph should be the same with the self.graph.'
        self.graph = self.graph + graph
    

    def load(self, fname: str):
        print(f'Load graph from {fname}.')
        with open(fname, 'rb') as f:
            that = pickle.load(f)
        for attr in ['item_map', 'num_items', 'directional', 'graph', 'pop_k', 'locale_sensitive', 'pop_items', 'id2pid']:
            setattr(self, attr, getattr(that, attr))
        
        


if __name__ == "__main__":
    import os

    data_type = 'all'
    work_dir = "/root/autodl-tmp/huangxu/Amazon-KDDCUP-23/"
    train_df = pd.read_csv(os.path.join(work_dir, f"data_for_recstudio/{data_type}_task_1_train_inter_feat.csv"))
    test_df = pd.read_csv(os.path.join(work_dir, f"data_for_recstudio/{data_type}_task_1_valid_inter_feat.csv"))
    product_df = pd.read_csv(os.path.join(work_dir, 'raw_data/products_train.csv'))
    item_map = { id: i+1 for i, id in enumerate(product_df['id'].unique()) }    # 0 saved for padding

    graph = CoOccuGraph(item_map)

    if not os.path.exists(f'./graph_{data_type}.gph'):
        graph.construct_graph(train_df,)
        graph.save(f'./graph_{data_type}.gph')
    else:
        graph.load(f'./graph_{data_type}.gph')

    test_graph = CoOccuGraph(item_map).construct_graph(test_df, True)

    graph.graph = graph.graph + test_graph

    pred = graph.predict(test_df, k=None, original_file=False)
    pred.to_parquet(f'./pred_{data_type}.parquet', engine='pyarrow')

    print("Test end.")
