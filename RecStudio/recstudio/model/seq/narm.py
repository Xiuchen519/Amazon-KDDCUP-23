from typing import List

import torch
from recstudio.data import dataset, advance_dataset
from recstudio.model import basemodel, loss_func, module, scorer
from recstudio.ann import sampler

r"""
Paper Reference:
##################
    Jing Li, et al. "Neural Attentive Session-based Recommendation" in CIKM 2017.
    https://dl.acm.org/doi/10.1145/3132847.3132926
"""


class NARMQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, embed_dim, hidden_size, layer_num, dropout_rate: List, item_encoder=None) -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.gru_layer = torch.nn.Sequential(
            self.item_encoder,
            torch.nn.Dropout(dropout_rate[0]),
            module.GRULayer(
                input_dim=embed_dim,
                output_dim=hidden_size,
                num_layer=layer_num,
            )
        )
        self.gather_layer = module.SeqPoolingLayer(pooling_type='last')
        self.attn_layer = module.AttentionLayer(q_dim=hidden_size, mlp_layers=[hidden_size], bias=False)

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate[1]),
            torch.nn.Linear(hidden_size*2, embed_dim, bias=False)
        )

    def forward(self, batch):
        gru_vec = self.gru_layer(batch['in_' + self.fiid])
        c_global = h_t = self.gather_layer(gru_vec, batch['seqlen'])  # B x H

        c_local = self.attn_layer(query=h_t.unsqueeze(1), key=gru_vec, value=gru_vec,
                                  key_padding_mask=batch['in_' + self.fiid] == 0, need_weight=False).squeeze(1)  # B x H

        c = torch.cat((c_global, c_local), dim=1)   # B x 2H
        query = self.fc(c)   # B x D
        return query


class NARM(basemodel.BaseRetriever):
    r""" NARM a hybrid encoder with an attention mechanism to model the user’s sequential behavior
    and capture the user’s main purpose in the current session, which are combined as a unified
    session representation later.

    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
        - ``hidden_size(int)``: The output size of GRU layer. Default: ``128``.
        - ``dropout_rate(list[float])``:  The dropout probablity of two dropout layers: the first
         | is after item embedding layer, the second is between the GRU layer and the bi-linear
         | similarity layer. Default: ``[0.25, 0.5]``.
        - ``layer_num(int)``: The number of layers for the GRU. Default: ``1``.
    """

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.item_fields = {train_data.fiid}
    
    def _set_data_field(self, data):
        data.use_field = set(
            [data.fuid, data.fiid, data.frating, 'locale', 
             'UK_index', 'DE_index', 'JP_index', 'ES_index', 'IT_index', 'FR_index']
        )

    def _get_dataset_class():
        r"""The dataset is SeqDataset."""
        return advance_dataset.KDDCUPSeqDataset


    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return NARMQueryEncoder(
            fiid=self.fiid,
            embed_dim=self.embed_dim,
            hidden_size=model_config['hidden_size'],
            layer_num=model_config['layer_num'],
            dropout_rate=model_config['dropout_rate'],
            item_encoder=self.item_encoder
        )

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        r"""Binary Cross Entropy is used as the loss function."""
        if self.config['model']['loss_func'] == 'softmax':
            return loss_func.SoftmaxLoss()
        elif self.config['model']['loss_func'] == 'sampled_softmax':
            return loss_func.SampledSoftmaxLoss()
        else:
            return loss_func.BinaryCrossEntropyLoss()

    def _get_sampler(self, train_data):
        r"""Uniform sampler is used as negative sampler."""
        if self.config['model']['loss_func'] == 'softmax':
            return None
        else:
            return sampler.UniformSampler(train_data.num_items)
