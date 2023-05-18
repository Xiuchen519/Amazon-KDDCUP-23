import torch
from recstudio.ann import sampler
from recstudio.data import dataset, advance_dataset
from recstudio.model.module import functional as recfn
from recstudio.model import basemodel, loss_func, module, scorer
from recstudio.model.module import MLPModule

import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_

class seqmlp_initialization(object):
    def __init__(self, initial_range=0.02) -> None:
        super().__init__()
        self.initial_range = initial_range

    def __call__(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initial_range)
            if module.padding_idx is not None:
                constant_(module.weight.data[module.padding_idx], 0.)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MLPQueryEncoder(torch.nn.Module):
    
    def __init__(
            self, config, fiid, max_seq_len, item_encoder) -> None:
        super().__init__()
        self.config = config
        self.fiid = fiid
        self.max_seq_len = max_seq_len
        self.item_encoder = item_encoder
        
        self.embed_dim = self.config['embed_dim']
        self.hidden_layers = config['hidden_layers']
        self.hidden_layers = [self.embed_dim * self.max_seq_len] + self.hidden_layers + [self.embed_dim]

        self.hidden_layers = MLPModule(
            self.hidden_layers,
            activation_func=config['activation'],
            dropout=config['dropout_rate'], 
            batch_norm=config['batch_norm'],
            last_activation=False)

            
    def forward(self, batch):
        batch_size = batch['in_'+self.fiid].shape[0]
        user_hist = batch['in_'+self.fiid] # [B, L]
        seq_item_index = torch.arange(self.max_seq_len, dtype=torch.long, device=user_hist.device).unsqueeze(dim=0) # [1, L]
        seq_item_index = seq_item_index.expand(user_hist.size(0), -1) # [B, L]
        seq_item_index = seq_item_index - (self.max_seq_len - batch['seqlen']).unsqueeze(dim=-1) # [B, L]
        padding_pos = seq_item_index < 0 
        seq_item_index[padding_pos] = 0
        reordered_use_hist = torch.gather(user_hist, dim=-1, index=seq_item_index) # [B, L]
        reordered_use_hist[padding_pos] = 0 # replace item on padding pos with 0 [B, L]
        
        seq_embs = self.item_encoder(reordered_use_hist).reshape(batch_size, -1) # [B, LxD]

        return self.hidden_layers(seq_embs) # [B, D]


class SeqMLP(basemodel.BaseRetriever):
    r"""
    Use MLP to model a user's sequence.

    """

    def _get_dataset_class():
        r"""SeqDataset is used for SeqMLP."""
        return advance_dataset.KDDCUPSeqDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.item_fields = {train_data.fiid}
    
    def _set_data_field(self, data):
        data.use_field = set(
            [data.fuid, data.fiid, data.frating, 'locale', 
             'UK_index', 'DE_index', 'JP_index', 'ES_index', 'IT_index', 'FR_index']
        )

    def _init_parameter(self):
        
        if self.config['train']['init_method'] == 'seqmlp_init':
            init_method = seqmlp_initialization(self.config['train']['init_range'])
            for name, module in self.named_children():
                if isinstance(module, basemodel.Recommender):
                    module._init_parameter()
                else:
                    module.apply(init_method)
        else:
            super()._init_parameter()


    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return MLPQueryEncoder(
            config=model_config,
            fiid=self.fiid,
            max_seq_len=train_data.config['max_seq_len'],
            item_encoder=self.item_encoder
        )

    def _get_item_encoder(self, train_data):
        emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        if self.config['train'].get("pretrained_embed_file", None) is not None:
            self.load_pretrained_embedding(emb, train_data, self.config['train']['pretrained_embed_file'])
        return emb

    def _get_score_func(self):
        r"""InnerProduct is used as the score function."""
        if self.config['model']['loss_func'] == 'ccl':
            return scorer.CosineScorer()
        else:
            return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""Binary Cross Entropy is used as the loss function."""
        if self.config['model']['loss_func'] == 'softmax':
            return loss_func.SoftmaxLoss()
        elif self.config['model']['loss_func'] == 'sampled_softmax':
            return loss_func.SampledSoftmaxLoss()
        elif self.config['model']['loss_func'] == 'ccl':
            return loss_func.CCLLoss(self.config['model']['negative_margin'], self.config['model']['negative_weight'])
        else:
            return loss_func.BinaryCrossEntropyLoss()

    def _get_sampler(self, train_data):
        r"""Uniform sampler is used as negative sampler."""
        if self.config['model']['loss_func'] == 'softmax':
            return None
        else:
            return sampler.UniformSampler(train_data.num_items) 

    
    def load_pretrained_embedding(self, embedding_layer, train_data, path: str):
        import pickle
        with open(path, 'rb') as f:
            emb_ckpt = pickle.load(f)
        
        self.logger.info(f"Load item embedding from {path}.")
        dataset_item_map = train_data.field2tokens[train_data.fiid]
        ckpt_item_map = emb_ckpt['map']
        id2id = torch.tensor([ckpt_item_map[i] if i!='[PAD]' else 0 for i in dataset_item_map ])
        emb = emb_ckpt['embedding'][id2id].contiguous().data

        assert emb.shape == embedding_layer.weight.shape
        embedding_layer.weight.data = emb
        torch.nn.init.constant_(embedding_layer.weight.data[embedding_layer.padding_idx], 0.)


    # def _get_sampler(self, train_data):
    #     r"""Uniform sampler is used as negative sampler."""
    #     return None

    # def candidate_topk(self, batch, k, user_h=None, return_query=False):
    #     self.item_vector = self.item_encoder(batch['last_item_candidates']) # [B, 300]
    #     score, topk_items = super().topk(batch, k, user_h, return_query) # [B, 150]
    #     topk_items = topk_items - 1 # [B, topk]
    #     topk_items = torch.gather(batch['last_item_candidates'], dim=-1, index=topk_items)
    #     return score, topk_items