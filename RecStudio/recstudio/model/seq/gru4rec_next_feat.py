from operator import mod
import torch
from recstudio.ann import sampler
from recstudio.data import dataset, advance_dataset
from recstudio.model import basemodel, loss_func, module, scorer
import logging 
import numpy as np 
from typing import Dict
from functools import partial

r"""
GRU4Rec
############

Paper Reference:
    Balazs Hidasi, et al. "Session-Based Recommendations with Recurrent Neural Networks" in ICLR2016.
    https://arxiv.org/abs/1511.06939
"""


class GRU4RecFeatItemEncoder(torch.nn.Module):

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

        if self.model_config['use_text_feat']:
            # item text features 
            self.item_text_vectors = torch.from_numpy(np.load(self.model_config['item_text_vectors'])).type(torch.float)
            self.item_text_emb = torch.nn.Embedding.from_pretrained(self.item_text_vectors, freeze=True, padding_idx=0)
            self.item_text_mlp = module.MLPModule(self.model_config['item_text_layers'], 'ReLU', dropout=self.model_config['item_text_dropout'], last_activation=False)

        if self.model_config['feat_id_concat']:
            self.feat_id_mlp = module.MLPModule(self.model_config['feat_id_layers'], 'ReLU', dropout=self.model_config['feat_id_dropout'], last_activation=False)

    
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
        item_text_embeddings = self.item_text_emb(item_ids) # [B, D2] or [B, L, D2]
        
        # if self.model_config['feat_id_concat']:
        #     return item_text_embeddings
        # else:
        #     return self.item_text_mlp(item_text_embeddings)
        return self.item_text_mlp(item_text_embeddings)

    def forward(self, batch):
        # st_time = time.time()
        # item categorical feature
        is_target = batch.get('is_target', True)
        item_cat_embeddings = self.get_cat_emb(batch, is_target=is_target) # [B, D] or [B, L, D]
        if self.model_config['use_text_feat']:
            item_text_embeddings = self.get_text_emb(batch, is_target=is_target)

        if is_target:
            item_id_embeddings = self.item_emb(batch[self.fiid]) # [B, D] or [B, L, D]
        else:
            item_id_embeddings = self.item_emb(batch['in_'+self.fiid]) # [B, D] or [B, L, D]
        # ed_time = time.time()
        # self.logger.info(f'encoder time : {ed_time - st_time}')
        if self.model_config['use_text_feat']:
            if self.model_config['feat_id_concat']:
                return self.feat_id_mlp(torch.concat([item_cat_embeddings, item_text_embeddings, item_id_embeddings], dim=-1))
            else:
                return item_cat_embeddings + item_text_embeddings + item_id_embeddings
        else:
            if self.model_config['feat_id_concat']:
                return self.feat_id_mlp(torch.concat([item_cat_embeddings, item_id_embeddings], dim=-1))
            else:
                return item_cat_embeddings + item_id_embeddings
            

class GRU4Rec_Next_Feat(basemodel.BaseRetriever):
    r"""
    GRU4Rec apply RNN in Recommendation System, where sequential behavior of user is regarded as input
    of the RNN.
    """

    # def add_model_specific_args(parent_parser):
    #     parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
    #     parent_parser.add_argument_group('GRU4Rec')
    #     parent_parser.add_argument("--hidden_size", type=int, default=128, help='hidden size of feedforward')
    #     parent_parser.add_argument("--layer_num", type=int, default=1, help='layer num of transformers')
    #     parent_parser.add_argument("--dropout_rate", type=float, default=0.2, help='dropout rate')
    #     parent_parser.add_argument("--negative_count", type=int, default=1, help='negative sampling numbers')
    #     return parent_parser
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.use_product_feature = self.config['model']['use_product_feature']

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
        data['is_target'] = False
        return data

    def _get_dataset_class():
        r"""SeqDataset is used for SASRec."""
        return advance_dataset.KDDCUPSeqDataset

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return (
            module.VStackLayer(
                module.HStackLayer(
                    torch.nn.Sequential(
                        self.item_encoder,
                        torch.nn.Dropout(model_config['dropout_rate']),
                        module.GRULayer(self.embed_dim, model_config['hidden_size'], model_config['layer_num']),
                    ),
                    module.LambdaLayer(lambda_func=lambda x: x['seqlen']),
                ),
                module.SeqPoolingLayer(pooling_type='last'),
                torch.nn.Linear(model_config['hidden_size'], self.embed_dim)
            )
        )

    def _get_item_encoder(self, train_data):
        model_config = self.config['model']
        return GRU4RecFeatItemEncoder(train_data, model_config, self.embed_dim, self.use_product_feature)

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
        
    
    def _get_item_vector(self):
        if len(self.item_fields) == 1 and isinstance(self.item_encoder, torch.nn.Embedding):
            return self.item_encoder.weight[1:]
        else:
            device = next(self.parameters()).device
            self.item_all_feat = self._to_device(self.item_all_feat, device)
            output = self.item_encoder(self.item_all_feat)
            return output[1:]
        
    def _get_ckpt_param(self):
        '''
        Returns:
            OrderedDict: the parameters to be saved as check point.
        '''
        ckpt_param = self.state_dict()
        if 'item_encoder.item_text_emb.weight' in ckpt_param:
            del ckpt_param['item_encoder.item_text_emb.weight']
        if 'query_encoder.item_encoder.item_text_emb.weight' in ckpt_param:
            del ckpt_param['query_encoder.item_encoder.item_text_emb.weight']
        return ckpt_param
    
    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self._parameter_device)

        self.config['model'] = ckpt['config']['model']
        if hasattr(self, '_update_item_vector'):
            self._update_item_vector()
            if 'item_vector' not in ckpt['parameters']:
                ckpt['parameters']['item_vector'] = self.item_vector
        self.load_state_dict(ckpt['parameters'], strict=False)
        
    
