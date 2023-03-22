import os 
import copy
import numpy as np 
import pandas as pd 
import torch
from torch.nn.utils.rnn import pad_sequence
from recstudio.data.dataset import TripletDataset, SeqDataset, SeqToSeqDataset, TensorFrame


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


class SessionDataset(SeqDataset):
    r"""Dataset for session-based recommendation.

    Not implemented now.
    """
    pass


class KDDCUPDataset(SeqToSeqDataset):


    # def _prepare_user_item_feat(self):
        # if self.user_feat is not None:
        #     self.user_feat.set_index(self.fuid, inplace=True)
        #     self.user_feat = self.user_feat.reindex(np.arange(self.num_users))
        #     self.user_feat.reset_index(inplace=True)
        #     self._fill_nan(self.user_feat, mapped=True)
        # else:
        #     self.user_feat = pd.DataFrame(
        #         {self.fuid: np.arange(self.num_users)})

        # if self.item_feat is not None:
        #     self.item_feat.set_index(self.fiid, inplace=True)
        #     self.item_feat = self.item_feat.reindex(np.arange(self.num_items))
        #     self.item_feat.reset_index(inplace=True)
        #     self._fill_nan(self.item_feat, mapped=True)
        # else:
        #     self.item_feat = pd.DataFrame(
        #         {self.fiid: np.arange(self.num_items)})
        # pass
            
    def _get_pos_data(self, index):
        # data_index : [user_id, start, end]
        # user interval [start, end) both including. 
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
            lens = end - start
            data['seqlen'] = lens
            l_source = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
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

    def build_test_dataset(self, test_data_path:str):

        test_inter_fields = ['sess_id:token', 'product_id:token', 'timestamp:float', 'locale:token']
        # load test feat
        test_inter_feat = super()._load_feat(
            feat_path=test_data_path, 
            header=0, 
            sep=',',
            feat_cols=test_inter_fields)
        
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
        
        test_dataset.data_index = self._get_data_idx((splits, uids))[0]
        user_hist, user_count = test_dataset.get_hist(True)
        test_dataset.user_hist = user_hist
        test_dataset.user_count = user_count
        
        return test_dataset
    
    def prediction_loader(self, batch_size, shuffle=False, num_workers=0, drop_last=False, ddp=False):
        r"""Return a dataloader for prediction"""
        self.eval_mode = True 
        self.predict_mode = True # set mode to prediction.
        return self.loader(batch_size, shuffle, num_workers, drop_last, ddp)



        
        