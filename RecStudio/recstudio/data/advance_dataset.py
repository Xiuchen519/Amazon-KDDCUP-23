from typing import List
import os 
import copy
import numpy as np 
import pandas as pd 
import torch
from torch.nn.utils.rnn import pad_sequence
from recstudio.data.dataset import TripletDataset, TensorFrame, SessionDataset, SessionSliceDataset, DataSampler
from torch.utils.data import DataLoader, Dataset, Sampler, DistributedSampler
from functools import partial

from tqdm import tqdm 
import logging 
import warnings
import scipy.sparse as ssp
from recstudio.utils import *
from typing import *
import pickle

from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from datasets import Dataset as TFDataset # transformer dataset 


class ALSDataset(TripletDataset):
    r"""ALSDataset is a dataset designed for Alternating Least Squares algorithms.

    For Alternating Least Squares algorithms, embeddings of users and items are optimized alternatively.
    So the data provided should be ``<u, Iu>`` and ``<i, Ui>`` alternatively.
    """

    def build(self, split_ratio, shuffle=True, split_mode='user_entry', **kwargs):
        datasets = self._build(split_ratio, shuffle, split_mode, True, False)
        data_index = datasets[0].inter_feat_subset
        user_ids = self.inter_feat.get_col(self.fuid)[data_index]
        user_uniq, count_train = torch.unique_consecutive(
            user_ids, return_counts=True)
        cumsum = torch.hstack([torch.tensor([0]), count_train.cumsum(-1)])
        datasets[0].data_index = torch.tensor(
            [[u, data_index[st], data_index[en-1]+1] for u, st, en in zip(user_uniq, cumsum[:-1], cumsum[1:])])
        return datasets

    def transpose(self):
        r"""Transpose user and item.

        The transpose operation will return a copy of the dataset after exchanging user and item.
        The returned dataset can easily return an item and all the interacted users, while the original
        dataset can only provide the user and its interacted items.

        Returns:
            recstudio.dataset.ALSDataset: the transposed dataset.
        """
        output = copy.copy(self)
        item_ids = self.inter_feat.get_col(self.fiid)
        data_index = self.inter_feat_subset
        indicator = torch.zeros_like(
            item_ids, dtype=torch.bool).scatter(0, data_index, True)
        sort_idx = (item_ids * 2 + ~indicator).sort().indices
        output.inter_feat = self.inter_feat.reindex(sort_idx)
        item_uniq, count_toal = torch.unique_consecutive(
            item_ids[sort_idx], return_counts=True)
        count_train = [_.sum() for _ in torch.split(
            indicator[sort_idx], tuple(count_toal))]
        cumsum = torch.hstack([torch.tensor([0]), count_toal.cumsum(-1)])
        output.data_index = torch.tensor(
            [[i, st, st+c] for i, st, c in zip(item_uniq, cumsum[:-1], count_train)])
        output.fuid = self.fiid
        output.fiid = self.fuid
        output.user_feat = self.item_feat
        output.item_feat = self.user_feat
        return output

    def save(self, file_name=None):
        import os

        import scipy.io as sio
        import scipy.sparse as ssp
        users, items, ratings = [], [], []
        for data in self.loader(batch_size=100, shuffle=True):
            uid, iid, rating = data[self.fuid], data[self.fiid], data[self.frating]
            for u, ids, rs in zip(uid, iid, rating):
                for id, r in zip(ids, rs):
                    if u > 0 and id > 0:
                        if 'user' in self.fuid:
                            users.append(u)
                            items.append(id)
                        else:
                            users.append(id)
                            items.append(u)
                        ratings.append(r)
        users = (torch.stack(users) - 1).numpy()
        items = (torch.stack(items) - 1).numpy()
        ratings = torch.stack(ratings).numpy()
        shape = [self.num_users-1, self.num_items-1]
        shape = shape if 'user' in self.fuid else shape.reverse()
        mat = ssp.csc_matrix((ratings, (users, items)), shape)
        #sio.savemat(os.path.join('datasets', file_name+'.mat'), {file_name:mat}, format='4')
        return mat


# class SessionDataset(SeqDataset):
#     r"""Dataset for session-based recommendation."""


class KDDCupCollator:

    def __init__(self, tokenizer) -> None:
        self.tokenizer:PreTrainedTokenizer = tokenizer
    
    def __call__(self, batch) -> Any:
        collated_batch = {}
        for feat_key in batch[0].keys():
            feat_list = []
            for i in range(len(batch)):
                feat_list.append(batch[i][feat_key])
            if isinstance(feat_list[0], torch.Tensor):
                feat_dim = feat_list[0].dim()
                if feat_dim == 0: # float, collate them into a tensor with length B
                    collated_batch[feat_key] = torch.tensor(feat_list)
                elif feat_dim == 1:
                    collated_batch[feat_key] = pad_sequence(feat_list, batch_first=True) # tensor, padding them to a same length
            elif isinstance(feat_list[0], BatchEncoding): # list[BatchEncoding] single sentence 
                collated_batch[feat_key] = self.tokenizer.pad(
                                            feat_list, # list[BatchEncoding]
                                            padding=True, 
                                            return_tensors='pt'
                                        )
            elif isinstance(feat_list[0], list) and isinstance(feat_list[0][0], BatchEncoding): #  list[list[BatchEncoding]] multi sentences 
                feat_list = sum(feat_list, [])
                collated_batch[feat_key] = self.tokenizer.pad(
                                            feat_list, # list[BatchEncoding]
                                            padding=True, 
                                            return_tensors='pt'
                                        )
        return collated_batch


class KDDCUPSeqDataset(SessionSliceDataset):
    
    def __init__(self, name, data_dir, config: Union[Dict, str] = None, cache=False):
    
        if cache == False:
            self.name = name

            self.logger = logging.getLogger('recstudio')
            self.config = config

            if 'tokenizer' in self.config:
                if isinstance(config['tokenizer'], str):
                    self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
                elif isinstance(config['tokenizer'], PreTrainedTokenizer):
                    self.tokenizer = config['tokenizer']

            self._init_common_field()
            self._load_all_data(data_dir, self.config['field_separator'])
            # first factorize user id and item id, and then filtering to
            # determine the valid user set and item set
            self._filter(self.config['min_user_inter'],
                            self.config['min_item_inter'])
            self._map_all_ids()
            self._post_preprocess()
            
            self._use_field = set([self.fuid, self.fiid, self.frating])
        

    @classmethod
    def build_datasets(cls, name: str = 'ml-100k', specific_config: Union[Dict, str] = None):

        def _load_cache(path):
            with open(path, 'rb') as f:
                download_obj = pickle.load(f)
            return download_obj

        def _save_cache(sub_datasets, md: str):
            cache_dir = os.path.join(DEFAULT_CACHE_DIR, "cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(os.path.join(cache_dir, md), 'wb') as f:
                pickle.dump(sub_datasets, f)

        logger = logging.getLogger('recstudio')

        config = get_dataset_default_config(name)
        if specific_config is not None:
            if isinstance(specific_config, str):
                config.update(parser_yaml(specific_config))
            elif isinstance(specific_config, Dict):
                config.update(specific_config)
            else:
                raise TypeError("expecting `config` to be Dict or string,"
                                f"while get {type(specific_config)} instead.")

        cache_flag, data_dir = check_valid_dataset(name, config)
        if cache_flag:
            logger.info(f"Load dataset from cache {data_dir}.")
            cache_datasets = _load_cache(data_dir)
            datasets = []
            for i in range(len(cache_datasets)):
                datasets.append(cls(name, data_dir, config, True))
                for k in cache_datasets[i].__dict__:
                    attr = getattr(cache_datasets[i], k)
                    setattr(datasets[i], k, attr)
            return datasets 
        else:
            dataset = cls(name, data_dir, config)
            # load train and valid dataset separately
            train_dataset = dataset.build(**config)[0] # split ratio : [1.0]
            valid_dataset = dataset.build_valid_dataset(data_dir, config['field_separator'])
            if config['save_cache'] == True:
                _save_cache([train_dataset, valid_dataset], md5(config))
            return [train_dataset, valid_dataset]
        

    def _filter(self, min_user_inter, min_item_inter):
        self._filter_ratings(self.config.get('low_rating_thres', None))
        item_list = self.inter_feat[self.fiid]
        item_idx_list, items = pd.factorize(item_list)
        user_list = self.inter_feat[self.fuid]
        user_idx_list, users = pd.factorize(user_list)
        warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
        user_item_mat = ssp.csc_matrix(
            (np.ones_like(user_idx_list), (user_idx_list, item_idx_list)))
        cols = np.arange(items.size)
        rows = np.arange(users.size)
        while(True): # TODO: only delete users/items in inter_feat, users/items in user/item_feat should also be deleted.
            m, n = user_item_mat.shape
            col_sum = np.squeeze(user_item_mat.sum(axis=0).A)
            col_ind = col_sum >= min_item_inter
            col_count = np.count_nonzero(col_ind)
            if col_count > 0:
                cols = cols[col_ind]
                user_item_mat = user_item_mat[:, col_ind]
            else:
                raise ValueError('All the interactions will be filtered, please adjust the min_item_inter.')
            row_sum = np.squeeze(user_item_mat.sum(axis=1).A)
            row_ind = row_sum >= min_user_inter
            row_count = np.count_nonzero(row_ind)
            if row_count > 0:
                rows = rows[row_ind]
                user_item_mat = user_item_mat[row_ind, :]
            else:
                raise ValueError('All the interactions will be filtered, please adjust the min_user_inter.')
            if col_count == n and row_count == m:
                break
            else:
                pass
        #
        keep_users = set(users[rows])
        keep_items = set(items[cols])
        keep = user_list.isin(keep_users)
        keep &= item_list.isin(keep_items)
        self.inter_feat = self.inter_feat[keep]
        self.inter_feat.reset_index(drop=True, inplace=True)
        # do not filter item or user feat even if they are not in inter_feat
        # if self.user_feat is not None:
        #    self.user_feat = self.user_feat[self.user_feat[self.fuid].isin(keep_users)]
        #    self.user_feat.reset_index(drop=True, inplace=True)
        # if self.item_feat is not None:
        #    self.item_feat = self.item_feat[self.item_feat[self.fiid].isin(keep_items)]
        #    self.item_feat.reset_index(drop=True, inplace=True)


    def _get_neg_data(self, data : Dict):
        neg_id = torch.randint(0, self.num_items, (self.neg_count,)).long() # [neg]
        neg_item_feat = self.item_feat[neg_id]
        # negatives should be flatten here.
        # After flatten and concat, the batch size will be B*(1+neg)
        for k, v in neg_item_feat.items():
            data['neg_' + k] = v # [neg] 

        if 'title' in self.use_field:
            locale_name = self.field2tokens['locale'][data['locale']]

            data['neg_title_input'] = self.title_feat[neg_item_feat[f'{locale_name}_index']]['input_ids'] # List[List]
            data['neg_title_input'] = [self.tokenizer.encode_plus(neg_title_ids, return_attention_mask=False, return_token_type_ids=False)
                                       for neg_title_ids in data['neg_title_input']]
            
        # delete index in data
        data_keys = list(data.keys())
        for k in data_keys:
            if 'index' in k:
                del data[k]

        return data


    def _get_pos_data(self, index):
        # [start, end] both included
        if getattr(self, 'predict_mode', False):
            idx = self.data_index[index] # [3]
            data = {self.fuid: idx[0]}
            data.update(self.user_feat[data[self.fuid]])
            start, end = idx[1], idx[2] 
            lens = end - start + 1
            data['seqlen'] = lens

            # source_data
            if not self.eval_mode:
                l_source = torch.arange(start, end + 1) # should include the last item.
            else:
                l_source = torch.arange(start, end) # when eval, the last item should be excluded.
            source_data = self.inter_feat[l_source] # [seq len]
            source_data.update(self.item_feat[source_data[self.fiid]])
            
            if self.config['item_candidates_path'] is not None:
                # candidate items data
                last_item = self.inter_feat[end][self.fiid]
                last_item_candidates = self.item_candidates_feat[last_item] 
                data['last_item_candidates'] = last_item_candidates['candidates'] # [300]

            if 'title' in self.use_field:
                locale_name = self.field2tokens['locale'][source_data['locale'][0]]

                source_data['title_input'] = self.title_feat[source_data[f'{locale_name}_index']]['input_ids'] # List[List]
                res_input_ids = source_data['title_input'][0] 
                for product_input_ids in source_data['title_input'][1 : ]:
                    res_input_ids.append(self.tokenizer.sep_token_id)
                    res_input_ids.extend(product_input_ids)
                source_data['title_input'] = res_input_ids # List[int]
                source_data['title_input'] = self.tokenizer.encode_plus(source_data['title_input'], return_attention_mask=False, return_token_type_ids=False) # BatchEncoding


            for k, v in source_data.items():
                if k != self.fuid:
                    data['in_' + k] = v

            # delete index in data
            data_keys = list(data.keys())
            for k in data_keys:
                if 'index' in k:
                    del data[k]

            return data
        else:
            idx = self.data_index[index]
            data = {self.fuid: idx[0]}
            data.update(self.user_feat[data[self.fuid]])
            start, end = idx[1], idx[2]
            lens = end - start
            data['seqlen'] = lens

            # get source data 
            l = torch.arange(start, end)
            source_data = self.inter_feat[l]
            source_data.update(self.item_feat[source_data[self.fiid]])

            # get target data 
            target_data = self.inter_feat[end]
            target_data.update(self.item_feat[target_data[self.fiid]])
            
            if self.config['item_candidates_path'] is not None:
                # candidate items data
                last_item = self.inter_feat[end - 1][self.fiid] # the last one is ground truth
                last_item_candidates = self.item_candidates_feat[last_item] 
                data['last_item_candidates'] = last_item_candidates['candidates'] # [300]
            
            if 'title' in self.use_field:
                locale_name = self.field2tokens['locale'][target_data['locale']]

                source_data['title_input'] = self.title_feat[source_data[f'{locale_name}_index']]['input_ids'] # List[List]
                res_input_ids = source_data['title_input'][0] 
                for product_input_ids in source_data['title_input'][1 : ]:
                    res_input_ids.append(self.tokenizer.sep_token_id)
                    res_input_ids.extend(product_input_ids)
                source_data['title_input'] = res_input_ids # List[int]
                
                target_data['title_input'] = self.title_feat[target_data[f'{locale_name}_index'].item()]['input_ids'] # List[int]

                source_data['title_input'] = self.tokenizer.encode_plus(source_data['title_input'], return_attention_mask=False, return_token_type_ids=False) # BatchEncoding
                target_data['title_input'] = self.tokenizer.encode_plus(target_data['title_input'], return_attention_mask=False, return_token_type_ids=False) # BatchEncoding
                
            for n, d in zip(['in_', ''], [source_data, target_data]):
                for k, v in d.items():
                    if k != self.fuid:
                        data[n+k] = v
            
            # delete index in data
            data_keys = list(data.keys())
            for k in data_keys:
                if 'index' in k:
                    del data[k]
            
            return data

    def _get_pos_data_2(self, index):
        # [start, end] both included
        if getattr(self, 'predict_mode', False):
            idx = self.data_index[index] # [B, 3]
            data = {self.fuid: idx[:, 0]}
            data.update(self.user_feat[data[self.fuid]])
            start = idx[:, 1] # [B]
            end = idx[:, 2]
            lens = end - start + 1
            data['seqlen'] = lens
            l_source = torch.cat([torch.arange(s, e + 1) for s, e in zip(start, end)]) # should include the last item.
            # source_data
            source_data = self.inter_feat[l_source]
            for k in source_data:
                source_data[k] = pad_sequence(source_data[k].split(
                    tuple(lens.numpy())), batch_first=True)
            source_data.update(self.item_feat[source_data[self.fiid]])
            
            if self.config['item_candidates_path'] is not None:
                # candidate items data
                last_items = self.inter_feat[end][self.fiid] # [B]
                last_item_candidates = self.item_candidates_feat[last_items] # [B, 300]
                data['last_item_candidates'] = last_item_candidates['candidates']

            for k, v in source_data.items():
                if k != self.fuid:
                    data['in_' + k] = v
            return data
        else:
            idx = self.data_index[index]
            data = {self.fuid: idx[:, 0]}
            data.update(self.user_feat[data[self.fuid]])
            target_data = self.inter_feat[idx[:, 2]]
            target_data.update(self.item_feat[target_data[self.fiid]])
            start = idx[:, 1]
            end = idx[:, 2]
            lens = end - start
            data['seqlen'] = lens
            l = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
            source_data = self.inter_feat[l]
            for k in source_data:
                source_data[k] = pad_sequence(source_data[k].split(
                    tuple(lens.numpy())), batch_first=True)
            source_data.update(self.item_feat[source_data[self.fiid]])

            if self.config['item_candidates_path'] is not None:
                # candidate items data
                last_items = self.inter_feat[end - 1][self.fiid] # the last one is ground truth
                last_item_candidates = self.item_candidates_feat[last_items]
                data['last_item_candidates'] = last_item_candidates['candidates']

            for n, d in zip(['in_', ''], [source_data, target_data]):
                for k, v in d.items():
                    if k != self.fuid:
                        data[n+k] = v
            return data
    

    def _prepare_user_item_feat(self):
        logger = logging.getLogger('recstudio')
        if self.user_feat is not None:
            self.user_feat.set_index(self.fuid, inplace=True)
            self.user_feat = self.user_feat.reindex(np.arange(self.num_users))
            self.user_feat.reset_index(inplace=True)
            self._fill_nan(self.user_feat, mapped=True)
        else:
            self.user_feat = pd.DataFrame(
                {self.fuid: np.arange(self.num_users)})

        if self.item_feat is not None:
            self.item_feat.index = self.item_feat.index + 1 
            self.item_feat = self.item_feat.reindex(np.arange(len(self.item_feat) + 1))
            self.item_feat.loc[0, self.fiid] = 0 # this row is for padding
            self._fill_nan(self.item_feat, mapped=True)
            
            item_index_data = {self.fiid : pd.Series(list(range(self.num_items)), dtype=pd.Int64Dtype())}
            for token in self.field2tokens['locale'][1:]:
                item_index_data[f'{token}_index'] = pd.Series([0] * self.num_items, dtype=pd.Int64Dtype())
            self.item_index_feat = pd.DataFrame(item_index_data)
            
            logger.info('start to create item index feat.')
            for locale_name in tqdm(self.field2tokens['locale'][1:]):
                locale_products = self.item_feat[self.item_feat['locale'] == self.field2token2idx['locale'][locale_name]]
                self.item_index_feat.loc[locale_products[self.fiid], f'{locale_name}_index'] = locale_products.index
                self.field2type[f'{locale_name}_index'] = 'int'
            logger.info('item index feat is ready.')
            
            if hasattr(self, 'tokenizer'):
                logger.info('********* start to create title feat ************')
                def tokenize_function(examples, tokenizer, max_length):
                    if 'title' in examples:
                        return tokenizer(examples['title'], 
                                        add_special_tokens=False, # don't add special tokens when preprocess
                                        truncation=True, 
                                        max_length=max_length,
                                        return_attention_mask=False,
                                        return_token_type_ids=False)
                not_title_columns = []
                for col in self.item_feat.columns:
                    if 'product_id' not in col and 'title' not in col and 'locale' not in col:
                        not_title_columns.append(col)
                self.title_feat = self.item_feat.drop(columns=not_title_columns) # no inplace, original item_feat is keeped.
                self.title_feat['title'][self.title_feat['title'] == ''] = self.tokenizer.unk_token
                self.title_feat = TFDataset.from_pandas(self.title_feat, preserve_index=False)
                self.title_feat = self.title_feat.map(partial(tokenize_function, tokenizer=self.tokenizer, max_length=self.config['max_title_len']), 
                                                    num_proc=8, remove_columns=["title"], batched=True)
                logger.info('********* title feat is ready ************')


            self.item_all_data = copy.deepcopy(self.item_feat)
            self.item_feat = self.item_index_feat
        else:
            self.item_feat = pd.DataFrame(
                {self.fiid: np.arange(self.num_items)})
            
        if self.config['item_candidates_path'] is not None:
            def map_token_2_id(token):
                if token in self.field2token2idx[self.fiid]:
                    return self.field2token2idx[self.fiid][token]
                else:
                    return 0

            logger.info('Start to process item_candidates!')
            
            if self.config['item_candidates_path'].endswith("ftr"):
                self.item_candidates_feat = pd.read_feather(self.config['item_candidates_path'])
            elif self.config['item_candidates_path'].endswith("parquet"):
                self.item_candidates_feat = pd.read_parquet(self.config['item_candidates_path'], engine='pyarrow')
            else:
                raise NotImplementedError("Only supported for ftr and parquet format file.")
            self.item_candidates_feat = self.item_candidates_feat[['id', 'candidates']]
            # item_ids = list(map(map_token_2_id, self.item_candidates_feat['id']))
            # item_ids = map((lambda x : self.field2token2idx[x]), self.item_candidates_feat['item'])
            self.item_candidates_feat['id'] = self.item_candidates_feat['id'].apply(map_token_2_id)
            self.item_candidates_feat['candidates'] = self.item_candidates_feat['candidates'].apply(lambda x: np.array([map_token_2_id(_) for _ in x]))
            
            # map item name in array into item id
            max_candidates_len = self.item_candidates_feat['candidates'].apply(len).max()

            # reset id 
            self.item_candidates_feat = self.item_candidates_feat.set_index('id')
            self.item_candidates_feat = self.item_candidates_feat.reindex(np.arange(len(self.item_candidates_feat) + 1)) # zero is for padding.
            
            # fill nan
            self.item_candidates_feat.iloc[0]['candidates'] = np.array([0] * max_candidates_len)

            # config field
            self.field2type['candidates'] = 'token_seq'
            self.field2maxlen['candidates'] = max_candidates_len

    def __getitem__(self, index):
        r"""Get data at specific index.

        Args:
            index(int): The data index.
        Returns:
            dict: A dict contains different feature.
        """
        data = self._get_pos_data(index)
        if (self.eval_mode or self.predict_mode) and 'user_hist' not in data:
            user_count = self.user_count[data[self.fuid]]
            data['user_hist'] = self.user_hist[data[self.fuid]][:user_count]
        else:
            # Negative sampling in dataset.
            # Only uniform sampling is supported now.
            if getattr(self, 'neg_count', None) is not None:
                if self.neg_count > 0:
                    data = self._get_neg_data(data)
        return data

    def dataframe2tensors(self):
        super().dataframe2tensors()
        if self.config['item_candidates_path'] is not None:
            self.item_candidates_feat = TensorFrame.fromPandasDF(self.item_candidates_feat, self)

    def _get_predict_data_idx(self, splits):
        splits, uids = splits
        maxlen = self.config['max_seq_len'] or (splits[:, -1] - splits[:, 0]).max()

        def get_slice(sps, uids):
            data = []
            for i, sp in enumerate(sps): # predict  
                if sp[1] > sp[0]:
                    slice_end = sp[1] - 1
                    slice_start = max(sp[0], sp[1] - maxlen)
                    data.append(np.array([uids[i], slice_start, slice_end]))
            return np.array(data)

        output = [get_slice(splits, uids)]
        output = [torch.from_numpy(_) for _ in output]
        return output

    def _get_valid_data_idx(self, splits):
        splits, uids = splits
        maxlen = self.config['max_seq_len'] or (splits[:, -1] - splits[:, 0]).max()

        def get_slice(sps, uids):
            data = []
            for i, sp in enumerate(sps): # predict  
                if sp[1] > sp[0]:
                    slice_end = sp[1] - 1
                    slice_start = max(sp[0], sp[1] - maxlen - 1) # the length should be maxlen + 1
                    data.append(np.array([uids[i], slice_start, slice_end]))
            return np.array(data)

        output = [get_slice(splits, uids)]
        output = [torch.from_numpy(_) for _ in output]
        return output

    def get_all_session_hist(self, isUser=True):
        r"""Get session history, include the last item.


        Args:
            isUser(bool, optional): Default: ``True``.

        Returns:
            torch.Tensor: padded user or item hisoty.

            torch.Tensor: length of the history sequence.
        """
        start = self.data_index[:, 1] # (user, [start, end])
        end = self.data_index[:, 2] # the last item is included in the history
        session_inter_feat_subset = torch.cat([torch.arange(s, e + 1, dtype=s.dtype) for s, e in zip(start, end)], dim=0)
        
        user_array = self.inter_feat.get_col(self.fuid)[session_inter_feat_subset]
        item_array = self.inter_feat.get_col(self.fiid)[session_inter_feat_subset]
        sorted, index = torch.sort(user_array if isUser else item_array)
        user_item, count = torch.unique_consecutive(sorted, return_counts=True)
        list_ = torch.split(
            item_array[index] if isUser else user_array[index], tuple(count.numpy()))
        tensors = [torch.tensor([], dtype=torch.int64) for _ in range(
            self.num_users if isUser else self.num_items)]
        for i, l in zip(user_item, list_):
            tensors[i] = l
        user_count = torch.tensor([len(e) for e in tensors])
        tensors = pad_sequence(tensors, batch_first=True)
        return tensors, user_count

    # load prediction dataset
    def build_test_dataset(self, test_data_path:str):

        test_inter_fields = ['sess_id:token', 'product_id:token', 'timestamp:float', 'locale:token']
        # load test feat
        test_inter_feat = super()._load_feat(
            feat_path=test_data_path, 
            header=0, 
            sep=',',
            feat_cols=test_inter_fields)
        
        # copy dataset to generate predict dataset 
        test_dataset = copy.copy(self)
        test_dataset.field2tokens = copy.copy(self.field2tokens)
        test_dataset.field2tokens['sess_id'] = np.concatenate([['[PAD]'], np.arange(test_inter_feat['sess_id'].astype('int').max() + 1)])
        test_dataset.field2token2idx['sess_id'] = {token : i for i, token in enumerate(test_dataset.field2tokens['sess_id'])}
        test_dataset.user_feat = pd.DataFrame(
                {self.fuid: np.arange(test_dataset.num_users)})

        # map ids 
        for inter_field in test_inter_fields:
            field_name, field_type = inter_field.split(':')[0], inter_field.split(':')[1]
            if 'float' in field_type:
                continue

            test_inter_feat[field_name] = test_inter_feat[field_name].map(
                lambda x : test_dataset.field2token2idx[field_name][x] if x in test_dataset.field2token2idx[field_name] else test_dataset.field2token2idx[field_name]['[PAD]']
                )
            
        # get splits and uids 
        test_inter_feat.sort_values(by=[self.fuid, self.ftime], inplace=True)
        user_count = test_inter_feat[self.fuid].groupby(
                test_inter_feat[self.fuid], sort=False).count()

        cumsum = user_count.cumsum()
        splits = cumsum.to_numpy()
        splits = np.concatenate([[0], splits])
        splits = np.array(list(zip(splits[:-1], splits[1:])))
        uids = user_count.index

        # transform test_inter_feat to TensorFrame 
        test_inter_feat = TensorFrame.fromPandasDF(test_inter_feat, self)
        test_dataset.user_feat = TensorFrame.fromPandasDF(test_dataset.user_feat, self)
        test_dataset.inter_feat = test_inter_feat
        
        test_dataset.data_index = self._get_predict_data_idx((splits, uids))[0]
        user_hist, user_count = test_dataset.get_all_session_hist(True)
        test_dataset.user_hist = user_hist
        test_dataset.user_count = user_count
        
        return test_dataset

    # load valid dataset
    def build_valid_dataset(self, data_dir, field_sep):

        # load valid feat
        valid_inter_feat_path = os.path.join(
            data_dir, self.config['valid_inter_feat_name'])
        valid_inter_feat = super()._load_feat(
            feat_path=valid_inter_feat_path, 
            header=self.config['inter_feat_header'], 
            sep=field_sep,
            feat_cols=self.config['inter_feat_field'])
        
        if self.frating is None:
            self.frating = 'rating'
            self.field2type[self.frating] = 'float'
            self.field2maxlen[self.frating] = 1
        if self.frating not in valid_inter_feat:
            # add ratings when implicit feedback
            valid_inter_feat.insert(0, self.frating, 1)
        
        # copy dataset to generate predict dataset 
        valid_dataset = copy.copy(self)

        # map ids 
        # user_feat, sess id as user id
        # Don't change the key in origin field2tokens, since field2tokens is shared. copy it first.   
        valid_dataset.field2tokens = copy.copy(self.field2tokens)
        valid_dataset.field2tokens['sess_id'] = np.concatenate([['[PAD]'], np.arange(valid_inter_feat['sess_id'].astype('int').max() + 1)])
        valid_dataset.field2token2idx['sess_id'] = {token : i for i, token in enumerate(valid_dataset.field2tokens['sess_id'])}
        valid_dataset.user_feat = pd.DataFrame(
                {self.fuid: np.arange(valid_dataset.num_users)})
        # inter_feat sess_id is also remapped here
        for inter_field in self.config['inter_feat_field']:
            field_name, field_type = inter_field.split(':')[0], inter_field.split(':')[1]
            if 'float' in field_type:
                continue

            valid_inter_feat[field_name] = valid_inter_feat[field_name].map(
                lambda x : valid_dataset.field2token2idx[field_name][x] if x in valid_dataset.field2token2idx[field_name] else valid_dataset.field2token2idx[field_name]['[PAD]']
                )
            
        # get splits and uids 
        valid_inter_feat.sort_values(by=[self.fuid, self.ftime], inplace=True)
        user_count = valid_inter_feat[self.fuid].groupby(
                valid_inter_feat[self.fuid], sort=False).count()

        cumsum = user_count.cumsum()
        splits = cumsum.to_numpy()
        splits = np.concatenate([[0], splits])
        splits = np.array(list(zip(splits[:-1], splits[1:])))
        uids = user_count.index

        # transform valid_inter_feat to TensorFrame 
        valid_inter_feat = TensorFrame.fromPandasDF(valid_inter_feat, self)
        valid_dataset.user_feat = TensorFrame.fromPandasDF(valid_dataset.user_feat, self)
        valid_dataset.inter_feat = valid_inter_feat
        
        valid_dataset.data_index = self._get_valid_data_idx((splits, uids))[0]
        user_hist, user_count = valid_dataset.get_eval_session_hist(True)
        valid_dataset.user_hist = user_hist
        valid_dataset.user_count = user_count
        
        return valid_dataset

    def get_locale_item_set(self, locale_name):
        item_all_feat_index = self.item_feat.get_col(f'{locale_name}_index')
        return torch.arange(self.num_items, dtype=torch.int)[item_all_feat_index != 0]

    def loader(self, batch_size, shuffle=True, num_workers=8, drop_last=False, ddp=False):
        # if not ddp:
        # Don't use SortedSampler here, it may hurt the performence of the model.
        if ddp:
            sampler = DistributedSampler(self, shuffle=shuffle)
            output = DataLoader(self, 
                                batch_size=batch_size,
                                sampler=sampler,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=KDDCupCollator(getattr(self, 'tokenizer', None)))
        else:
            output = DataLoader(self, 
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                collate_fn=KDDCupCollator(getattr(self, 'tokenizer', None)))
        return output
    
    def train_loader(self, batch_size, shuffle=True, num_workers=8, drop_last=False, ddp=False):
        return super().train_loader(batch_size, shuffle, num_workers, drop_last, ddp)

    def prediction_loader(self, batch_size, shuffle=False, num_workers=8, drop_last=False, ddp=False):
        r"""Return a dataloader for prediction"""
        self.eval_mode = True 
        self.predict_mode = True # set mode to prediction.
        return self.loader(batch_size, False, num_workers, drop_last, ddp)

    def eval_loader(self, batch_size, num_workers=8, ddp=False):
        self.eval_mode = True
        return self.loader(batch_size, shuffle=False, num_workers=num_workers, ddp=ddp)


class KDDCUPSessionDataset(KDDCUPSeqDataset): # don't cut session into slice 

    def _get_data_idx(self, splits):
        # split: [start, train_end, valid_end, test_end]
        splits, uids = splits 
        maxlen = self.config['max_seq_len'] or (splits[:, -1] - splits[:, 0] - 1).max()

        def get_slice(sps, uids, sp_idx):  
            data = []
            for i, sp in enumerate(sps): # val or test 
                if sp[sp_idx] > sp[sp_idx - 1]:
                    slice_end = sp[sp_idx] - 1
                    slice_start = max(sp[sp_idx - 1], sp[sp_idx] - 1 - maxlen)
                    data.append(np.array([uids[i], slice_start, slice_end]))
            return np.array(data)

        output = [get_slice(splits, uids, i) for i in range(1, splits.shape[-1])]
        output = [torch.from_numpy(_) for _ in output]
        return output

    @property
    def inter_feat_subset(self):
        """self.data_index : [num_users, 3]
        The intervel in data_index is both closed.
        data_index only includes interactions in the truncated sequence of a user, instead of all interactions.
        Return:
            torch.tensor: the history index in inter_feat. shape: [num_interactions_in_train]
        """
        start = self.data_index[:, 1]
        end = self.data_index[:, 2]
        return torch.cat([torch.arange(s, e + 1, dtype=s.dtype) for s, e in zip(start, end)], dim=0)
    


class KDDCUPDataset(SessionDataset):
# class KDDCUPDataset(SeqToSeqDataset):

    def _filter(self, min_user_inter, min_item_inter):
        self._filter_ratings(self.config.get('low_rating_thres', None))
        item_list = self.inter_feat[self.fiid]
        item_idx_list, items = pd.factorize(item_list)
        user_list = self.inter_feat[self.fuid]
        user_idx_list, users = pd.factorize(user_list)
        warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
        user_item_mat = ssp.csc_matrix(
            (np.ones_like(user_idx_list), (user_idx_list, item_idx_list)))
        cols = np.arange(items.size)
        rows = np.arange(users.size)
        while(True): # TODO: only delete users/items in inter_feat, users/items in user/item_feat should also be deleted.
            m, n = user_item_mat.shape
            col_sum = np.squeeze(user_item_mat.sum(axis=0).A)
            col_ind = col_sum >= min_item_inter
            col_count = np.count_nonzero(col_ind)
            if col_count > 0:
                cols = cols[col_ind]
                user_item_mat = user_item_mat[:, col_ind]
            else:
                raise ValueError('All the interactions will be filtered, please adjust the min_item_inter.')
            row_sum = np.squeeze(user_item_mat.sum(axis=1).A)
            row_ind = row_sum >= min_user_inter
            row_count = np.count_nonzero(row_ind)
            if row_count > 0:
                rows = rows[row_ind]
                user_item_mat = user_item_mat[row_ind, :]
            else:
                raise ValueError('All the interactions will be filtered, please adjust the min_user_inter.')
            if col_count == n and row_count == m:
                break
            else:
                pass
        #
        keep_users = set(users[rows])
        keep_items = set(items[cols])
        keep = user_list.isin(keep_users)
        keep &= item_list.isin(keep_items)
        self.inter_feat = self.inter_feat[keep]
        self.inter_feat.reset_index(drop=True, inplace=True)
        # do not filter item or user feat even if they are not in inter_feat
        # if self.user_feat is not None:
        #    self.user_feat = self.user_feat[self.user_feat[self.fuid].isin(keep_users)]
        #    self.user_feat.reset_index(drop=True, inplace=True)
        # if self.item_feat is not None:
        #    self.item_feat = self.item_feat[self.item_feat[self.fiid].isin(keep_items)]
        #    self.item_feat.reset_index(drop=True, inplace=True)

    def _prepare_user_item_feat(self):
        if self.user_feat is not None:
            self.user_feat.set_index(self.fuid, inplace=True)
            self.user_feat = self.user_feat.reindex(np.arange(self.num_users))
            self.user_feat.reset_index(inplace=True)
            self._fill_nan(self.user_feat, mapped=True)
        else:
            self.user_feat = pd.DataFrame(
                {self.fuid: np.arange(self.num_users)})

        if self.item_feat is not None:
            self.item_feat.index = self.item_feat.index + 1 
            self.item_feat = self.item_feat.reindex(np.arange(len(self.item_feat) + 1))
            self.item_feat.loc[0, self.fiid] = 0 # this row is for padding
            self._fill_nan(self.item_feat, mapped=True)
            
            item_index_data = {self.fiid : pd.Series(list(range(self.num_items)), dtype=pd.Int64Dtype())}
            for token in self.field2tokens['locale'][1:]:
                item_index_data[f'{token}_index'] = pd.Series([0] * self.num_items, dtype=pd.Int64Dtype())
            self.item_index_feat = pd.DataFrame(item_index_data)
            
            logger = logging.getLogger('recstudio')
            logger.info('start to create item index feat.')
            # for i in tqdm(range(len(self.item_feat))):
            #     product = self.item_feat.iloc[i]
            #     product_id, product_locale_id = product[self.fiid], product['locale']
            #     if product_id != 0 and product_locale_id != 0:
            #         product_locale_name = self.field2tokens['locale'][product_locale_id]
            #         self.item_index_feat.loc[product_id, f'{product_locale_name}_index'] = i
            for locale_name in tqdm(self.field2tokens['locale'][1:]):
                locale_products = self.item_feat[self.item_feat['locale'] == self.field2token2idx['locale'][locale_name]]
                self.item_index_feat.loc[locale_products[self.fiid], f'{locale_name}_index'] = locale_products.index
                self.field2type[f'{locale_name}_index'] = 'int'

            self.item_all_data = copy.deepcopy(self.item_feat)
            self.item_feat = self.item_index_feat
        else:
            self.item_feat = pd.DataFrame(
                {self.fiid: np.arange(self.num_items)})
            
    def _get_pos_data(self, index):
        # data_index : [user_id, start, end]
        # user interval [start, end] both including. 
        # training:
        # source: interval [idx[:, 1], idx[:, 2] - 1]
        # target: interval [idx[:, 1] + 1, idx[:, 2]]
        # valid/test:
        # source: interval [idx[:, 1], idx[:, 2] - 1]
        # target: idx[:, 2]
        
        if getattr(self, 'predict_mode', False):
            idx = self.data_index[index]
            data = {self.fuid: idx[:, 0]}
            data.update(self.user_feat[data[self.fuid]])
            start = idx[:, 1]
            end = idx[:, 2]
            lens = end - start + 1
            data['seqlen'] = lens
            l_source = torch.cat([torch.arange(s, e + 1) for s, e in zip(start, end)]) # to include the last item.
            # source_data
            source_data = self.inter_feat[l_source]
            for k in source_data:
                source_data[k] = pad_sequence(source_data[k].split(
                    tuple(lens.numpy())), batch_first=True)
            source_data.update(self.item_feat[source_data[self.fiid]])
            
            for k, v in source_data.items():
                if k != self.fuid:
                    data['in_' + k] = v
            return data
        else:
            return super()._get_pos_data(index)


    def _get_predict_data_idx(self, splits):
        splits, uids = splits
        maxlen = self.config['max_seq_len'] or (splits[:, -1] - splits[:, 0] - 1).max()

        def get_slice(sps, uids):
            data = []
            for i, sp in enumerate(sps): # predict  
                if sp[1] > sp[0]:
                    slice_end = sp[1] - 1
                    slice_start = max(sp[0], sp[1] - maxlen)
                    data.append(np.array([uids[i], slice_start, slice_end]))
            return np.array(data)

        output = [get_slice(splits, uids)]
        output = [torch.from_numpy(_) for _ in output]
        return output


    # load prediction dataset 
    def build_test_dataset(self, test_data_path:str):

        test_inter_fields = ['sess_id:token', 'product_id:token', 'timestamp:float', 'locale:token']
        # load test feat
        test_inter_feat = super()._load_feat(
            feat_path=test_data_path, 
            header=0, 
            sep=',',
            feat_cols=test_inter_fields)
        
        # copy dataset to generate predict dataset 
        test_dataset = copy.copy(self)
        test_dataset.field2tokens['sess_id'] = np.concatenate([['[PAD]'], np.arange(test_inter_feat['sess_id'].astype('int').max() + 1)])
        test_dataset.field2token2idx['sess_id'] = {token : i for i, token in enumerate(test_dataset.field2tokens['sess_id'])}
        test_dataset.user_feat = pd.DataFrame(
                {self.fuid: np.arange(test_dataset.num_users)})

        # map ids 
        for inter_field in test_inter_fields:
            field_name, field_type = inter_field.split(':')[0], inter_field.split(':')[1]
            if 'float' in field_type:
                continue

            test_inter_feat[field_name] = test_inter_feat[field_name].map(
                lambda x : test_dataset.field2token2idx[field_name][x] if x in test_dataset.field2token2idx[field_name] else test_dataset.field2token2idx[field_name]['[PAD]']
                )
            
        # get splits and uids 
        test_inter_feat.sort_values(by=[self.fuid, self.ftime], inplace=True)
        user_count = test_inter_feat[self.fuid].groupby(
                test_inter_feat[self.fuid], sort=False).count()

        cumsum = user_count.cumsum()
        splits = cumsum.to_numpy()
        splits = np.concatenate([[0], splits])
        splits = np.array(list(zip(splits[:-1], splits[1:])))
        uids = user_count.index

        # transform test_inter_feat to TensorFrame 

        test_inter_feat = TensorFrame.fromPandasDF(test_inter_feat, self)
        test_dataset.user_feat = TensorFrame.fromPandasDF(test_dataset.user_feat, self)
        test_dataset.inter_feat = test_inter_feat
        
        test_dataset.data_index = self._get_predict_data_idx((splits, uids))[0]
        user_hist, user_count = test_dataset.get_hist(True)
        test_dataset.user_hist = user_hist
        test_dataset.user_count = user_count
        
        return test_dataset

    def get_locale_item_set(self, locale_name):
        item_all_feat_index = self.item_feat.get_col(f'{locale_name}_index')
        return torch.arange(self.num_items, dtype=torch.int)[item_all_feat_index != 0]

    
    def prediction_loader(self, batch_size, shuffle=False, num_workers=0, drop_last=False, ddp=False):
        r"""Return a dataloader for prediction"""
        self.eval_mode = True 
        self.predict_mode = True # set mode to prediction.
        return self.loader(batch_size, shuffle, num_workers, drop_last, ddp)



        
        