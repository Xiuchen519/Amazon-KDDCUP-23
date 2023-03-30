import os 
import copy
import numpy as np 
import pandas as pd 
import torch
from torch.nn.utils.rnn import pad_sequence
from recstudio.data.dataset import TripletDataset, SeqDataset, SeqToSeqDataset, TensorFrame, SessionDataset, SessionSliceDataset
from tqdm import tqdm 
import logging 
import warnings
import scipy.sparse as ssp



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

class KDDCUPSeqDataset(SessionSliceDataset):

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

    def _get_pos_data(self, index):
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
            
        if self.config['item_candidates_path'] is not None:
            def map_token_2_id(token):
                if token in self.field2token2idx[self.fiid]:
                    return self.field2token2idx[self.fiid][token]
                else:
                    return 0

            logger.info('Start to process item_candidates!')
            
            self.item_candidates_feat = pd.read_feather(self.config['item_candidates_path'])
            item_ids = list(map(map_token_2_id, self.item_candidates_feat['id']))
            # item_ids = map((lambda x : self.field2token2idx[x]), self.item_candidates_feat['item'])
            self.item_candidates_feat['id'] = item_ids 
            
            # map item name in array into item id
            candidates_id_list = []
            for i in tqdm(range(len(self.item_candidates_feat))):
                candidates_id = np.array(list(map(map_token_2_id, self.item_candidates_feat.iloc[i]['candidates'])))
                # candidates_id = map((lambda x : self.field2token2idx[x]), self.item_candidates_feat.iloc[i]['candidates'])
                candidates_id_list.append(candidates_id)
            self.item_candidates_feat['candidates'] = candidates_id_list

            # reset id 
            self.item_candidates_feat = self.item_candidates_feat.set_index('id')
            self.item_candidates_feat = self.item_candidates_feat.reindex(np.arange(len(self.item_candidates_feat) + 1)) # zero is for padding.
            
            # fill nan
            self.item_candidates_feat.iloc[0]['candidates'] = np.array([0] * 300)

            # config field
            self.field2type['candidates'] = 'token_seq'
            self.field2maxlen['candidates'] = 500

    def dataframe2tensors(self):
        super().dataframe2tensors()
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

    def get_all_session_hist(self, isUser=True):
        r"""Get session history, exclude the last item.


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

    def get_locale_item_set(self, locale_name):
        item_all_feat_index = self.item_feat.get_col(f'{locale_name}_index')
        return torch.arange(self.num_items, dtype=torch.int)[item_all_feat_index != 0]
    
    def prediction_loader(self, batch_size, shuffle=False, num_workers=0, drop_last=False, ddp=False):
        r"""Return a dataloader for prediction"""
        self.eval_mode = True 
        self.predict_mode = True # set mode to prediction.
        return self.loader(batch_size, shuffle, num_workers, drop_last, ddp)

    


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



        
        