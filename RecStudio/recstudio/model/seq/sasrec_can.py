import torch
from recstudio.ann import sampler
from recstudio.data import dataset, advance_dataset
from recstudio.model.module import functional as recfn
from recstudio.model import basemodel, loss_func, module, scorer
from typing import Dict

class SASRecQueryEncoder(torch.nn.Module):
    def __init__(
            self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder,
            bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.bidirectional = bidirectional
        self.training_pooling_type = training_pooling_type
        self.eval_pooling_type = eval_pooling_type
        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.training_pooling_layer = module.SeqPoolingLayer(pooling_type=self.training_pooling_type)
        self.eval_pooling_layer = module.SeqPoolingLayer(pooling_type=self.eval_pooling_type)

    def forward(self, batch, need_pooling=True):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_encoder(user_hist)

        mask4padding = user_hist == 0  # BxL
        L = user_hist.size(-1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD

        if need_pooling:
            if self.training:
                if self.training_pooling_type == 'mask':
                    return self.training_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                if self.eval_pooling_type == 'mask':
                    return self.eval_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.eval_pooling_layer(transformer_out, batch['seqlen'])
        else:
            return transformer_out


class SASRec_CAN(basemodel.BaseRetriever):
    r"""
    SASRec models user's sequence with a Transformer.

    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
        - ``hidden_size(int)``: The output size of Transformer layer. Default: ``128``.
        - ``layer_num(int)``: The number of layers for the Transformer. Default: ``2``.
        - ``dropout_rate(float)``:  The dropout probablity for dropout layers after item embedding
         | and in Transformer layer. Default: ``0.5``.
        - ``head_num(int)``: The number of heads for MultiHeadAttention in Transformer. Default: ``2``.
        - ``activation(str)``: The activation function in transformer. Default: ``"gelu"``.
        - ``layer_norm_eps``: The layer norm epsilon in transformer. Default: ``1e-12``.
    """

    # def add_model_specific_args(parent_parser):
    #     parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
    #     parent_parser.add_argument_group('SASRec')
    #     parent_parser.add_argument("--hidden_size", type=int, default=128, help='hidden size of feedforward')
    #     parent_parser.add_argument("--layer_num", type=int, default=2, help='layer num of transformers')
    #     parent_parser.add_argument("--head_num", type=int, default=2, help='head num of multi-head attention')
    #     parent_parser.add_argument("--dropout_rate", type=float, default=0.5, help='dropout rate')
    #     parent_parser.add_argument("--negative_count", type=int, default=1, help='negative sampling numbers')
    #     return parent_parser

    def _get_dataset_class():
        r"""SeqDataset is used for SASRec."""
        return advance_dataset.KDDCUPSeqDataset
    
    def _set_data_field(self, data):
        if self.config['model']['loss_func'] != 'CanSoftmax':
            data.use_field = set([data.fuid, data.fiid, data.frating, 'locale'])
        else:
            data.use_field = set([data.fuid, data.fiid, data.frating, 'locale', 'candidates'])

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return SASRecQueryEncoder(
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=model_config['head_num'],
            hidden_size=model_config['hidden_size'], dropout=model_config['dropout_rate'],
            activation=model_config['activation'], layer_norm_eps=model_config['layer_norm_eps'],
            n_layer=model_config['layer_num'],
            item_encoder=self.item_encoder
        )

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        r"""InnerProduct is used as the score function."""
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""Binary Cross Entropy is used as the loss function."""
        if self.config['model']['loss_func'] == 'Softmax' or self.config['model']['loss_func'] == 'CanSoftmax':
            return loss_func.SoftmaxLoss()
        elif self.config['model']['loss_func'] == "BCE":
            return loss_func.BinaryCrossEntropyLoss()


    def _get_sampler(self, train_data):
        r"""Uniform sampler is used as negative sampler."""
        if self.config['model']['loss_func'] in ['Softmax', 'CanSoftmax']:
            return None
        else:
            return sampler.UniformSampler(train_data.num_items) 

    # def _get_sampler(self, train_data):
    #     r"""Uniform sampler is used as negative sampler."""
    #     return None

    def forward(self, batch: Dict, full_score: bool = False, return_query: bool = False, return_item: bool = False, return_neg_item: bool = False, return_neg_id: bool = False):
        if self.config['model']['loss_func'] == 'CanSoftmax':
            output = {}
            pos_items = self._get_item_feat(batch)
            pos_item_vec = self.item_encoder(pos_items)

            query = self.query_encoder(self._get_query_feat(batch))
            pos_score = self.score_func(query, pos_item_vec) # [B]
            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf')  # padding
            output['score'] = {'pos_score': pos_score}
            
            item_vectors = self.item_encoder(batch['last_item_candidates']) # [B, 300, 128]
            all_item_scores = self.score_func(query, item_vectors) # [B, 300]
            # include self
            all_item_scores = torch.cat([all_item_scores, pos_score.unsqueeze(dim=-1)], dim=-1)
            
            output['score']['all_score'] = all_item_scores 
            return output
        else:
            return super().forward(batch, full_score, return_query, return_item, return_neg_item, return_neg_id)

    def topk(self, batch, k, user_h=None, return_query=False):
        self.item_vector = self.item_encoder(batch['last_item_candidates']) # [B, 300]
        score, topk_items = super().topk(batch, k, user_h, return_query) # [B, 150]
        topk_items = topk_items - 1 # [B, topk]
        topk_items = torch.gather(batch['last_item_candidates'], dim=-1, index=topk_items)
        return score, topk_items


