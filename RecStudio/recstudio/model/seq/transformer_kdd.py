import torch
import pandas as pd
from collections import OrderedDict
from recstudio.data import advance_dataset
from recstudio.model import basemodel, loss_func
from recstudio.model.module.layers import AttentionLayer, MLPModule, SeqPoolingLayer

from recstudio.ann.sampler import UniformSampler



class Transformer_KDD(basemodel.BaseRanker):


    def _set_data_field(self, data):
        data.use_field = set([data.fuid, data.fiid, data.frating, 'locale'])


    def _get_dataset_class():
        r"""The dataset is SeqDataset."""
        return advance_dataset.KDDCUPSeqDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_conf = self.config['model']
        self.item_embedding = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        self.position_embedding = torch.nn.Embedding(train_data.config['max_seq_len'] + 2, self.embed_dim, padding_idx=0)

        trm_enc_layer = torch.nn.TransformerEncoderLayer(
            self.embed_dim, nhead=model_conf['nhead'], dim_feedforward=model_conf['dim_feedforward'], 
            dropout=model_conf['dropout'], activation=model_conf['activation'], batch_first=True)
        self.transformers = torch.nn.TransformerEncoder(trm_enc_layer, num_layers=model_conf['num_layers'])
        self.pooling_layer = SeqPoolingLayer('last')

        if model_conf['mlp_layer'] is None:
            self.mlp = None
            layer_size = [self.embed_dim]
        else:
            layer_size = [self.embed_dim] + model_conf['mlp_layer']
            self.mlp = MLPModule(layer_size, activation_func='ReLU', dropout=model_conf['dropout'])
        
        self.pred = torch.nn.Sequential(OrderedDict({
            'linear': torch.nn.Linear(layer_size[-1], 1),
            'sigmoid': torch.nn.Sigmoid()
        }))

        self.sampler = UniformSampler(train_data.num_items)

    def score(self, batch):
        item_seq = batch["in_" + self.fiid]
        seq_len = batch['seqlen'] 
        item_seq = torch.cat((item_seq, item_seq.new_zeros(item_seq.size(0),1)), dim=-1)
        item_seq[ torch.arange(item_seq.size(0)), seq_len] = batch[self.fiid]
        seq_len = seq_len + 1

        # positions = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        # positions = positions.unsqueeze(0).expand_as(item_seq)
        positions = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device).unsqueeze(dim=0) # [1, L]
        positions = positions.expand(item_seq.size(0), -1) # [B, L]
        padding_pos = positions >= seq_len.unsqueeze(dim=-1) # [B, L]
        positions = seq_len.unsqueeze(dim=-1) - positions # [B, L]
        positions[padding_pos] = 0
        position_embs = self.position_embedding(positions)

        seq_emb = self.item_embedding(item_seq)
        padding_mask = item_seq == 0

        tfm_out = self.transformers(src=seq_emb+position_embs, src_key_padding_mask=padding_mask)
        out = self.pooling_layer(tfm_out, batch['seqlen'])

        if self.mlp is not None:
            out = self.mlp(out)

        score = self.pred(out)
        return score.squeeze()  # B,

    def _get_loss_func(self):
        return loss_func.BCELoss()
    

    def forward(self, batch):
        num_sample = self.config['train']['num_candidates']
        all_candidates = batch['last_item_candidates']
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
    

    def predict_step(self, batch, dataset):
        topk = self.config['eval']['predict_topk']
        topk = min(topk, dataset.field2maxlen['candidates'])
        candidates = batch['last_item_candidates']
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
    
