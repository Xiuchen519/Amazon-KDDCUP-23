import torch
from typing import Dict
from recstudio.ann import sampler
from recstudio.data import dataset, advance_dataset
from recstudio.model.module import functional as recfn
from recstudio.model import basemodel, loss_func, module, scorer
import time
import logging 
import numpy as np 


class SASRecQueryEncoder(torch.nn.Module):
    def __init__(
            self, train_data, model_config, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder,
            bidirectional=False, training_pooling_type='last', eval_pooling_type='last', use_product_feature=False) -> None:
        super().__init__()
        self.fiid = fiid
        self.use_fields = train_data.use_field
        self.model_config = model_config
        self.item_encoder : SASRecFeatItemEncoder = item_encoder
        self.bidirectional = bidirectional
        self.training_pooling_type = training_pooling_type
        self.eval_pooling_type = eval_pooling_type
        self.use_product_feature = use_product_feature
        self.position_emb = torch.nn.Embedding(max_seq_len + 1, embed_dim if use_product_feature else embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim if use_product_feature else embed_dim,
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

        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device).unsqueeze(dim=0) # [1, L]
        positions = positions.expand(user_hist.size(0), -1) # [B, L]
        padding_pos = positions >= batch['seqlen'].unsqueeze(dim=-1) # [B, L]
        positions = batch['seqlen'].unsqueeze(dim=-1) - positions # [B, L]
        positions[padding_pos] = 0
        position_embs = self.position_emb(positions)
        
        seq_embs = self.item_encoder(batch, is_target=False)
        input_embs = seq_embs + position_embs # [B, L, D]

        mask4padding = user_hist == 0  # BxL
        L = user_hist.size(-1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        transformer_out = self.transformer_layer(
            src=self.dropout(input_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD

        if need_pooling:
            if self.training:
                if self.training_pooling_type == 'mask':
                    transformer_out = self.training_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    transformer_out = self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                if self.eval_pooling_type == 'mask':
                    transformer_out = self.eval_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    transformer_out = self.eval_pooling_layer(transformer_out, batch['seqlen'])
        else:
            transformer_out = transformer_out
        
        return transformer_out


class SASRecFeatItemEncoder(torch.nn.Module):

    def __init__(self, train_data, model_config, embed_dim, use_product_feature=True) -> None:
        super().__init__()
        self.logger = logging.getLogger('recstudio')
        self.fiid = train_data.fiid
        self.use_fields = train_data.use_field
        self.model_config = model_config
        self.use_product_feature = use_product_feature

        self.item_emb = torch.nn.Embedding(train_data.num_items, embed_dim, padding_idx=0)

        # item categorical features  
        if self.use_product_feature:
            # price embedding
            if self.model_config['use_price']:
                self.linear_price = torch.nn.Linear(1, embed_dim, bias=False)
                # self.norm_price = torch.nn.BatchNorm1d(embed_dim)
                self.activate_price = torch.nn.ReLU()
                self.linear_price_2 = torch.nn.Linear(embed_dim, embed_dim, bias=False)

            # brand embedding 
            self.brand_embedding = torch.nn.Embedding(train_data.num_values('brand'), embed_dim, padding_idx=0)
            # material embedding 
            self.material_embedding = torch.nn.Embedding(train_data.num_values('material'), embed_dim, padding_idx=0)
            # author embedding 
            self.author_embedding = torch.nn.Embedding(train_data.num_values('author'), embed_dim, padding_idx=0)
            # color embedding 
            if self.model_config['use_color']:
                self.color_embedding = torch.nn.Embedding(train_data.num_values('color'), embed_dim, padding_idx=0)

        # item text features 
        self.item_title_vectors = torch.from_numpy(np.load(self.config['model']['item_title_vectors'])).type(torch.float)
        self.item_title_emb = torch.nn.Embedding.from_pretrained(self.item_title_vectors, freeze=True, padding_idx=0)
        self.item_title_mlp = module.MLPModule(self.model_config['model']['item_title_layers'], 'ReLU', dropout=self.model_config['model']['item_title_dropout'], last_activation=False)

        if self.model_config['feat_id_concat']:
            self.feat_id_mlp = module.MLPModule(self.model_config['model']['feat_id_layers'], 'ReLU', dropout=self.model_config['model']['feat_id_dropout'], last_activation=False)

    
    def get_cat_emb(self, batch, is_target=True):
        if not is_target:
            brand_emb = self.brand_embedding(batch['in_brand']) # [B, L, D]
            material_emb = self.material_embedding(batch['in_material']) # [B, L, D]
            author_emb = self.author_embedding(batch['in_author']) # [B, L, D]
            feature_emb = brand_emb + material_emb + author_emb # [B, L, D]                    

            if 'price' in self.use_fields:
                price_emb = self.linear_price(batch['in_price'].unsqueeze(dim=-1)) # [B, L, D]
                price_emb = self.activate_price(price_emb)
                price_emb = self.linear_price_2(price_emb)
                feature_emb += price_emb

            # st_time = time.time()
            if 'color' in self.use_fields:
                color_seq = batch['in_color'] # [B, L, N]
                color_num_seq = batch['in_color_num'] # [B, L]
                color_num_seq[color_num_seq == 0] = 1
                color_emb = self.color_embedding(color_seq).sum(dim=-2) / color_num_seq.unsqueeze(dim=-1) # [B, L, N, D] -> [B, L, D]
                feature_emb += color_emb 
            # ed_time = time.time()
            # self.logger.info(f'color time : {ed_time - st_time}')


        else:
            brand_emb = self.brand_embedding(batch['brand']) # [B, D]
            material_emb = self.material_embedding(batch['material']) # [B, D]
            author_emb = self.author_embedding(batch['author']) # [B, D]
            feature_emb = brand_emb + material_emb + author_emb

            if 'price' in self.use_fields:
                price_emb = self.linear_price(batch['price'].unsqueeze(dim=-1)) # [B, D]
                price_emb = self.activate_price(self.norm_price(price_emb)) # [B, D]
                feature_emb += price_emb

            # st_time = time.time()
            if 'color' in self.use_fields:
                color_seq = batch['color'] # [B, N]
                if 'color_num' in batch:
                    color_num_seq = batch['color_num'] # [B]
                else:
                    color_num_seq = (batch['color'] != 0).sum(dim=-1) # [B]
                color_num_seq[color_num_seq == 0] = 1
                color_emb = self.color_embedding(color_seq).sum(dim=-2) / color_num_seq.unsqueeze(dim=-1) # [B, N, D] -> [B, D]
                feature_emb += color_emb  
            # ed_time = time.time()
            # self.logger.info(f'color time : {ed_time - st_time}')

        return feature_emb
    

    def get_text_emb(self, batch, is_target=True):
        if is_target:
            item_ids = batch[self.fiid]
        else:
            item_ids = batch['in_'+self.fiid]
        item_title_embeddings = self.item_title_emb(item_ids) # [B, D2] or [B, L, D2]
        
        if self.model_config['feat_id_concat']:
            return item_title_embeddings
        else:
            return self.item_title_mlp(item_title_embeddings)


    def forward(self, batch, is_target=True):
        # st_time = time.time()
        # item categorical feature
        item_cat_embeddings = self.get_cat_emb(batch, is_target=is_target) # [B, D] or [B, L, D]
        item_text_embeddings = self.get_text_emb(batch, is_target=is_target)

        if is_target:
            item_id_embeddings = self.item_emb(batch[self.fiid]) # [B, D] or [B, L, D]
        else:
            item_id_embeddings = self.item_emb(batch['in_'+self.fiid]) # [B, D] or [B, L, D]
        # ed_time = time.time()
        # self.logger.info(f'encoder time : {ed_time - st_time}')
        if self.model_config['feature_id_concat']:
            return self.feat_id_mlp(torch.concat([item_cat_embeddings, item_text_embeddings, item_id_embeddings], dim=-1))
        else:
            return item_cat_embeddings + item_text_embeddings + item_id_embeddings


class SASRec_Next_Feat(basemodel.BaseRetriever):
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
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.use_product_feature = self.config['model']['use_product_feature']

    def _get_dataset_class():
        r"""SeqDataset is used for SASRec."""
        return advance_dataset.KDDCUPSeqDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.num_items = train_data.num_items 
        if self.use_product_feature:
            use_field = set([train_data.fiid, 'brand', 'material', 'author'])
            if self.config['model']['use_color']:
                use_field.add('color')
            if self.config['model']['use_price']:
                use_field.add('price')
            self.item_fields = use_field
        else:
            self.item_fields = {train_data.fiid} # only use product id as item feature, other product features are used in query feature.
        
        self.item_all_data = train_data.item_all_data
        item_feat_data = self.item_feat.data
        for k in item_feat_data.keys():
            if 'index' in k:
                idx_name = k
        self.item_all_feat = self.item_all_data[item_feat_data[idx_name].type(torch.long)]
        self.item_all_feat = self._get_item_feat(self.item_all_feat)
        
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

    def _get_query_feat(self, data):
        return data

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return SASRecQueryEncoder(
            train_data=train_data, model_config=model_config,
            fiid=self.fiid, embed_dim=self.embed_dim,
            max_seq_len=train_data.config['max_seq_len'], n_head=model_config['head_num'],
            hidden_size=model_config['hidden_size'], dropout=model_config['dropout_rate'],
            activation=model_config['activation'], layer_norm_eps=model_config['layer_norm_eps'],
            n_layer=model_config['layer_num'],
            item_encoder=self.item_encoder,
            use_product_feature=self.use_product_feature
        )

    def _get_item_encoder(self, train_data):
        model_config = self.config['model']
        return SASRecFeatItemEncoder(train_data, model_config, self.embed_dim, self.use_product_feature)

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
        
    def _get_item_vector(self):

        # return self.item_encoder.item_emb.weight[1:]

        if len(self.item_fields) == 1 and isinstance(self.item_encoder, torch.nn.Embedding):
            return self.item_encoder.weight[1:]
        else:
            device = next(self.parameters()).device
            self.item_all_feat = self._to_device(self.item_all_feat, device)
            
            # st_time = time.time()
            # self.item_feat_data = self.item_feat.data
            # for k in self.item_feat_data.keys():
            #     if 'index' in k:
            #         idx_name = k
            # item_all_feat = self.item_all_data[self.item_feat_data[idx_name].type(torch.long)]
            # en_time = time.time()
            # self.logger.info(f'item_all_feat time : {en_time - st_time}s')

            # st_time = time.time()
            output = self.item_encoder(self.item_all_feat)
            # en_time = time.time()
            # self.logger.info(f'output time : {en_time - st_time}s')
            return output[1:]
        

    def _update_item_vector(self):  # TODO: update frequency setting
        # st_time = time.time()
        item_vector = self._get_item_vector()
        # en_time = time.time()
        # self.logger.info(f'item vector time : {en_time - st_time}s')
        if not hasattr(self, "item_vector"):
            self.register_buffer('item_vector', item_vector.detach().clone() if isinstance(item_vector, torch.Tensor) \
                else item_vector.copy())
        else:
            self.item_vector = item_vector

        if self.use_index:
            self.ann_index = self.build_ann_index()

    
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

    # def candidate_topk(self, batch, k, user_h=None, return_query=False):
    #     self.item_vector = self.item_encoder(batch['last_item_candidates']) # [B, 300]
    #     score, topk_items = super().topk(batch, k, user_h, return_query) # [B, 150]
    #     topk_items = topk_items - 1 # [B, topk]
    #     topk_items = torch.gather(batch['last_item_candidates'], dim=-1, index=topk_items)
    #     return score, topk_items