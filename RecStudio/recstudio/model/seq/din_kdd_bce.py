import torch
from typing import Dict
from recstudio.data import dataset, advance_dataset
from recstudio.model import basemodel, loss_func
from recstudio.model.module.layers import AttentionLayer, MLPModule

from recstudio.ann.sampler import UniformSampler
import pandas as pd

import recstudio.eval as eval


r"""
DIN
######################

Paper Reference:
    Guorui Zhou, et al. "Deep Interest Network for Click-Through Rate Prediction" in KDD2018.
    https://dl.acm.org/doi/10.1145/3219819.3219823

"""


class DIN_KDD_BCE(basemodel.BaseRanker):
    r"""
        | Deep Interest Network (DIN) designs a local activation unit to adaptively learn the representation
          of user interests from historical behaviors with respect to a certain ad.

        | DIN calculate the relevance between the target item and items in the sequence by adapting an
          attention machnism. But the method could not be applied to recall on all items in prediction
          due to the huge time cost.
    """
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.use_product_feature = self.config['model']['use_product_feature']

    def _set_data_field(self, data):
        if self.use_product_feature:
            use_field = set([data.fuid, data.fiid, data.frating, 'locale', 'brand', 'material', 'author',
                             'UK_index', 'JP_index', 'DE_index', 'ES_index', 'FR_index', 'IT_index'])
            if self.config['model']['use_color']:
                use_field.add('color')
            if self.config['model']['use_price']:
                use_field.add('price')
            data.use_field = use_field
        else:
            data.use_field = set([data.fuid, data.fiid, data.frating, 'locale',
                                  'UK_index', 'JP_index', 'DE_index', 'ES_index', 'FR_index', 'IT_index'])

    def _get_dataset_class():
        r"""The dataset is SeqDataset."""
        return advance_dataset.KDDCUPSeqDataset
    

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.item_all_data = train_data.item_all_data
        self.locale_field2tokens = train_data.field2tokens['locale']

        d = self.embed_dim
        model_config = self.config['model']
        act_f = model_config['activation']
        fc_mlp = model_config['fc_mlp']
        dropout_p = model_config['dropout']
        self.item_embedding = torch.nn.Embedding(train_data.num_items, d, 0)
        self.item_bias = torch.nn.Embedding(train_data.num_items, 1, padding_idx=0)
        if self.use_product_feature:
            self.activation_unit = AttentionLayer(
                2*d, 6*d, mlp_layers=model_config['attention_mlp'], activation=act_f) # the position of key and query is wrong.
            norm = [torch.nn.BatchNorm1d(2*d)] if model_config['batch_norm'] else []
            norm.append(torch.nn.Linear(2*d, 2*d))
            self.norm = torch.nn.Sequential(*norm)
            self.dense_mlp = MLPModule(
                [6*d]+fc_mlp, activation_func=act_f, dropout=dropout_p, batch_norm=model_config['batch_norm'])
        else:
            self.activation_unit = AttentionLayer(
                d, 3*d, mlp_layers=model_config['attention_mlp'], activation=act_f) # the position of key and query is wrong.
            norm = [torch.nn.BatchNorm1d(d)] if model_config['batch_norm'] else []
            norm.append(torch.nn.Linear(d, d))
            self.norm = torch.nn.Sequential(*norm)
            self.dense_mlp = MLPModule(
                [3*d]+fc_mlp, activation_func=act_f, dropout=dropout_p, batch_norm=model_config['batch_norm'])
        
        self.fc = torch.nn.Linear(fc_mlp[-1], 1)

        self.sampler = UniformSampler(train_data.num_items)

        if self.use_product_feature:
            # price embedding
            self.linear_price = torch.nn.Linear(1, self.embed_dim, bias=False)
            self.norm_price = torch.nn.BatchNorm1d(self.embed_dim)
            self.activate_price = torch.nn.ReLU()
            # brand embedding 
            self.brand_embedding = torch.nn.Embedding(train_data.num_values('brand'), self.embed_dim, padding_idx=0)
            # material embedding 
            self.material_embedding = torch.nn.Embedding(train_data.num_values('material'), self.embed_dim, padding_idx=0)
            # author embedding 
            self.author_embedding = torch.nn.Embedding(train_data.num_values('author'), self.embed_dim, padding_idx=0)
            # color embedding 
            self.color_embedding = torch.nn.Embedding(train_data.num_values('color'), self.embed_dim, padding_idx=0)

    def get_feature_emb(self, batch, is_target=False):
        if not is_target:
            brand_emb = self.brand_embedding(batch['in_brand']) # [B, L, D]
            material_emb = self.material_embedding(batch['in_material']) # [B, L, D]
            author_emb = self.author_embedding(batch['in_author']) # [B, L, D]
            feature_emb = brand_emb + material_emb + author_emb # [B, L, D]                    

            if 'price' in self.fields:
                price_emb = self.linear_price(batch['in_price'].unsqueeze(dim=-1)) # [B, L, D]
                price_emb = self.activate_price(self.norm_price(price_emb.transpose(-2, -1))).transpose(-2, -1)
                feature_emb += price_emb

            if 'color' in self.fields:
                color_seq = batch['in_color'] # [B, L, N]
                color_num_seq = batch['in_color_num'] # [B, L]
                color_num_seq[color_num_seq == 0] = 1
                color_emb = self.color_embedding(color_seq).sum(dim=-2) / color_num_seq.unsqueeze(dim=-1) # [B, L, N, D] -> [B, L, D]
                feature_emb += color_emb 

        else:
            brand_emb = self.brand_embedding(batch['brand']) # [B, D]
            material_emb = self.material_embedding(batch['material']) # [B, D]
            author_emb = self.author_embedding(batch['author']) # [B, D]
            feature_emb = brand_emb + material_emb + author_emb

            if 'price' in self.fields:
                price_emb = self.linear_price(batch['price'].unsqueeze(dim=-1)) # [B, D]
                price_emb = self.activate_price(self.norm_price(price_emb)) # [B, D]
                feature_emb += price_emb

            if 'color' in self.fields:
                color_seq = batch['color'] # [B, N]
                color_num_seq = batch['color_num'] # [B]
                color_num_seq[color_num_seq == 0] = 1
                color_emb = self.color_embedding(color_seq).sum(dim=-2) / color_num_seq.unsqueeze(dim=-1) # [B, N, D] -> [B, D]
                feature_emb += color_emb  

        return feature_emb

    def score(self, batch):
        seq_emb = self.item_embedding(batch['in_'+self.fiid]) # [B, L, D]
        target_emb = self.item_embedding(batch[self.fiid]) # [B, D]
        item_bias = self.item_bias(batch[self.fiid]).squeeze(-1)

        if self.use_product_feature:
            # concat feature embedding 
            seq_feature_emb = self.get_feature_emb(batch, is_target=False) # [B, L, D]
            target_feature_emb = self.get_feature_emb(batch, is_target=True) # [B, D]
            seq_emb = torch.cat([seq_emb, seq_feature_emb], dim=-1) # [B, L, 2 * D]
            target_emb = torch.cat([target_emb, target_feature_emb], dim=-1) # [B, 2 * D]

        target_emb_ = target_emb.unsqueeze(1).repeat(1, seq_emb.size(1), 1)   # BxLxD
        attn_seq = self.activation_unit(
            query=target_emb.unsqueeze(1),
            key=torch.cat((seq_emb, target_emb_*seq_emb, target_emb_-seq_emb), dim=-1), # fix target_emb_ -> seq_emb
            value=seq_emb,
            key_padding_mask=(batch['in_'+self.fiid] == 0),
            softmax=False
        ).squeeze(1)
        attn_seq = self.norm(attn_seq)
        cat_emb = torch.cat((attn_seq, target_emb, target_emb*attn_seq), dim=-1)
        score = self.fc(self.dense_mlp(cat_emb)).squeeze(-1)
        return score + item_bias

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()

    # def _get_loss_func(self):
    #     return torch.nn.CrossEntropyLoss()
    
    def forward(self, batch, test=False):
        num_sample = self.config['train']['num_candidates']
        all_candidates = batch['last_item_candidates']
        if test:    # for validation and test
            candidates = all_candidates[:, :self.config['eval']['predict_topk']].contiguous()
            all_scores = self.score_multi(batch, candidates) # [B, N]
            all_labels = (candidates == batch[self.fiid].view(-1,1)).float() # [B, N]
        else:
            if self.config['train']['candidate_strategy'] == 'cand':
                num_candidates = all_candidates.size(1)
                rand_idx = torch.randint(0, num_candidates, size=(all_candidates.size(0), num_sample), device=all_candidates.device)
                candidates = torch.gather(all_candidates, -1, rand_idx)
            elif self.config['train']['candidate_strategy'] == 'cand+rand':
                num_candidates = all_candidates.size(1)
                rand_idx = torch.randint(0, num_candidates, size=(all_candidates.size(0), num_sample // 2), device=all_candidates.device)
                candidates = torch.gather(all_candidates, -1, rand_idx)
                uni_cand = self.sampler.forward(all_candidates.size(0), num_sample - (num_sample // 2))[0].to(all_candidates.device)
                candidates = torch.cat([candidates, uni_cand], dim=-1)
            elif self.config['train']['candidate_strategy'] == 'rand':
                candidates = self.sampler.forward(all_candidates.size(0), num_sample)
            else:
                raise ValueError("Not supported for such strategy.")

            label_candidates = (candidates == batch[self.fiid].view(-1,1)).float()
            pos_score = self.score(batch)
            cand_scores = self.score_multi(batch, candidates)
            
            all_scores = torch.cat([pos_score.view(-1,1), cand_scores], dim=1)  # B, 1+len(candidates)
            all_labels = torch.cat([label_candidates.new_ones((label_candidates.size(0), 1)), label_candidates], dim=1)

        return {'pos_score': all_scores.view(-1), 'label': all_labels.view(-1)}, None

    # def forward(self, batch, test=False):
    #     num_sample = self.config['train']['num_candidates']
    #     all_candidates = batch['last_item_candidates'][:, :num_sample].contiguous()
        
    #     if test:
    #         candidates = all_candidates
    #         labels = (all_candidates == batch[self.fiid].view(-1, 1)).float() # [B, N_CAND]
    #     else:        
    #         if self.config['train']['candidate_strategy'] == 'cand':
    #             candidates = all_candidates

    #         valid_sample_idx = torch.nonzero(candidates == batch[self.fiid].view(-1, 1))[:, 0]
    #         labels = torch.nonzero(candidates == batch[self.fiid].view(-1, 1))[:, 1] # [B]
            
    #         candidates = candidates[valid_sample_idx].contiguous()
    #         for k in batch.keys():
    #             batch[k] = batch[k][valid_sample_idx].contiguous()

    #     cand_scores = self.score_multi(batch, candidates) # [B, N_CAND]
        
    #     return {'pos_score': cand_scores, 'label': labels}, None
    
    def training_step(self, batch):
        y_h, output = self.forward(batch)
        loss = self.loss_fn(**y_h) # + loss using output
        return loss

    def predict_step(self, batch, dataset, with_score):
        num_sample = self.config['train']['num_candidates']
        candidates = batch['last_item_candidates'][:, :num_sample].contiguous()
        
        topk = self.config['eval']['predict_topk']
        topk = min(topk, num_sample)
        
        
        cand_scores = self.score_multi(batch, candidates)

        scores, _idx = torch.topk(cand_scores, k=topk, dim=-1)
        topk_items = torch.gather(candidates, -1, _idx)        

        topk_item_tokens = []
        for topk_items_u in topk_items:
            tokens = dataset.field2tokens[self.fiid][topk_items_u.cpu()].tolist()
            topk_item_tokens.append(tokens)
        
        # turn locale id to locale name 
        locales_ids = batch['in_' + 'locale']
        locale_tokens = [dataset.field2tokens['locale'][locale_id[0]] for locale_id in locales_ids]

        prediction_df = pd.DataFrame({'locale' : locale_tokens, 'next_item_prediction' : topk_item_tokens})
        return prediction_df


    def _test_step(self, batch, metric, cutoffs=None):
        rank_m = eval.get_rank_metrics(metric)
        pred_m = eval.get_pred_metrics(metric) # 从pred_metrics中删除掉AUC
        bs = batch[self.frating].size(0)
        if len(rank_m) > 0:
            assert cutoffs is not None, 'expected cutoffs for topk ranking metrics.'
        # TODO: discuss in which cases pred_metrics should be calculated. According to whether there are neg labels in dataset?
        # When there are neg labels in dataset, should rank_metrics be considered?
        if self.retriever is None:
            result, _ = self.forward(batch, test=True)
            global_m = eval.get_global_metrics(metric)
            metrics = {}
            for n, f in pred_m:
                if n == 'gauc':
                    pred = result['pos_score'].view(bs, -1)
                    label = result['label'].view(bs, -1)
                    gauc = 0.0
                    non_zero_cnt = 0
                    for i in range(bs):
                        cur_gauc = eval.auc(pred[i], label[i])
                        if cur_gauc != 0:
                            gauc += cur_gauc
                            non_zero_cnt += 1
                    metrics[n] = gauc / non_zero_cnt
                else:
                    if not n in global_m:
                        metrics[n] = f(result['pos_score'], result['label'])
            if len(global_m) > 0:
                # gather scores and labels for global metrics like AUC.
                metrics['score'] = result['pos_score'].detach()
                metrics['label'] = result['label']
            
            if len(rank_m) > 0:
                pred = result['pos_score'].reshape(bs, -1)
                label = result['label'].reshape(bs, -1)
                sorted_socre, indices = torch.sort(pred, dim=-1, descending=True)
                sorted_label = torch.gather(label, dim=-1, index=indices)
                for cutoff in cutoffs:
                    for name, func in rank_m:
                        metrics[f'{name}@{cutoff}'] = func(sorted_label, sorted_socre, cutoff)    

        else:
            # pair-wise, support topk-based metrics, like [NDCG, Recall, Precision, MRR, MAP, MR, et al.]
            # The case is suitable for the scene where there are only positives in dataset.
            topk = self.config['eval']['topk']
            score, topk_items = self.topk(batch, topk, batch['user_hist'])
            if batch[self.fiid].dim() > 1:
                target, _ = batch[self.fiid].sort()
                idx_ = torch.searchsorted(target, topk_items)
                idx_[idx_ == target.size(1)] = target.size(1) - 1
                label = torch.gather(target, 1, idx_) == topk_items
                pos_rating = batch[self.frating]
            else:
                label = batch[self.fiid].view(-1, 1) == topk_items
                pos_rating = batch[self.frating].view(-1, 1)
            metrics = {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
        return metrics, bs

    def _generate_multi_item_batch(self, batch, item_ids):
        num_item = item_ids.size(-1)
        item_ids = item_ids.view(-1)

        locale_index = batch['in_locale'][:, 0] # [B]
        multi_locale_index = locale_index.unsqueeze(1) \
            .expand(-1, num_item, *tuple([-1 for i in range(len(locale_index.shape)-1)])) # [B, 1]
        multi_locale_index = multi_locale_index.reshape(-1, *(locale_index.shape[1:]))
        items = self._get_item_feat(item_ids, multi_locale_index)
        if isinstance(items, torch.Tensor): # only id
            items = {self.fiid: items}  # convert to dict
        multi_item_batch = {}
        #
        for k, v in batch.items():
            if (k not in items):
                multi_item_batch[k] = v.unsqueeze(1) \
                                       .expand(-1, num_item, *tuple([-1 for i in range(len(v.shape)-1)]))
                multi_item_batch[k] = multi_item_batch[k].reshape(-1, *(v.shape[1:]))
            else:
                multi_item_batch[k] = items[k]
        
        for k, v in items.items():
            if k not in multi_item_batch:
                multi_item_batch[k] = v
                
        return multi_item_batch

    def _get_item_feat(self, data, locale_index=None):
        if isinstance(data, dict):  # batch_data
            if len(self.item_fields) == 1:
                return data[self.fiid]
            else:
                return dict((field, value) for field, value in data.items() if field in self.item_fields)
        else:  # neg_item_idx
            if len(self.item_fields) == 1:
                return data
            else:
                assert locale_index is not None, 'locale index is needed to get the full features of items.'
                all_item_index = torch.zeros(len(data), dtype=torch.int64)
                for i, locale_name in enumerate(self.locale_field2tokens):
                    if locale_name == '[PAD]':
                        all_item_index[locale_index == i] = 0
                    else:
                        all_item_index[locale_index == i] = self.item_feat[data[locale_index == i]][f'{locale_name}_index'].to(torch.int64)
                item_all_data = self._to_device(self.item_all_data[all_item_index], self._parameter_device)
                if 'color' in item_all_data:
                    item_all_data['color_num'] = (item_all_data['color'] != 0).sum(dim=-1)
                return item_all_data