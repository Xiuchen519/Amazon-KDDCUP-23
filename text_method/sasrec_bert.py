import sys 
sys.path = ['./RecStudio'] + sys.path
import os 
from typing import Dict, Optional
import logging 
from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel, TrainingArguments
from recstudio.model.scorer import InnerProductScorer
from recstudio.model.loss_func import SoftmaxLoss
from recstudio.model import module
from arguments import ModelArguments
from transformers.file_utils import ModelOutput


logger = logging.getLogger(__name__)

@dataclass
class EncoderOutput(ModelOutput): # if a attribute is None, then it will not be in .items().
    query_reps: Optional[Tensor] = None
    item_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class SASRecQueryEncoder(torch.nn.Module):
    def __init__(
            self, config, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer,
            bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
        super().__init__()
        self.fiid = fiid
        self.bidirectional = bidirectional
        self.training_pooling_type = training_pooling_type
        self.eval_pooling_type = eval_pooling_type
        
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
        
        if self.reverse_pos:
            positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device).unsqueeze(dim=0) # [1, L]
            positions = positions.expand(user_hist.size(0), -1) # [B, L]
            padding_pos = positions >= batch['seqlen'].unsqueeze(dim=-1) # [B, L]
            positions = batch['seqlen'].unsqueeze(dim=-1) - positions # [B, L]
            positions[padding_pos] = 0
        else:
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

class SASRec_Bert(nn.Module):
    TRANSFORMER_CLS = AutoModel
    
    def __init__(self,
                 args,
                 train_data, 
                 bert : PreTrainedModel, 
                 sentence_pooling_method : str = 'cls', 
                 negatives_x_device : bool = False) -> None:
        super().__init__()
        self.fiid = train_data.fiid

        self.bert : PreTrainedModel = bert

        self.item_embeddings = torch.nn.Embedding(train_data.num_items, args.embed_dim, padding_idx=0)
        self.item_transform = torch.nn.Linear(768 + args.embed_dim, args.item_dim, bias=True)

        # transformer
        self.position_emb = torch.nn.Embedding(train_data.max_seq_len + 1, args.item_dim, padding_idx=0) # plus one for padding
        self.bidirectional = args.bidirectional
        self.training_pooling_type = args.training_pooling_type
        self.eval_pooling_type = args.eval_pooling_type
        
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=args.embed_dim,
            nhead=args.n_head,
            dim_feedforward=args.hidden_size,
            dropout=args.dropout,
            activation=args.activation,
            layer_norm_eps=args.layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=args.n_layer,
        )
        self.dropout = torch.nn.Dropout(p=args.dropout)
        self.training_pooling_layer = module.SeqPoolingLayer(pooling_type=self.training_pooling_type)
        self.eval_pooling_layer = module.SeqPoolingLayer(pooling_type=self.eval_pooling_type)


        self.loss_func = SoftmaxLoss()
        self.scorer = InnerProductScorer()

        self.sentence_pooling_method = sentence_pooling_method
        self.negatives_x_device = negatives_x_device

        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()


    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
    
    def encode_product_title(self, batch):
        # product_title_input: [N, L_t]
        # item_id: [B, L_seq]

        if batch['product_title_input'] is None: # for predict
            return None 
        
        product_title_out = self.bert(**batch['product_title_input'], return_dict=True) # [N, L_t]
        product_title_hidden = product_title_out.last_hidden_state
        product_title_reps = self.sentence_embedding(product_title_hidden, batch['product_title_input']['attention_mask'])
        return product_title_reps.contiguous()


    def encode_query(self, batch):
        # product_title_input: [N, L_t]
        # item_id: [B, L_seq]
        user_hist = batch['in_'+self.fiid]
        seqlen = batch['seqlen'] # [B]
        batch_size = user_hist.shape[0]

        # get sequence embeddings
        product_title_reps = self.encode_product_title(batch) # [N, D_bert]
        
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device).unsqueeze(dim=0) # [1, L]
        positions = positions.expand(user_hist.size(0), -1) # [B, L]
        padding_pos = positions >= batch['seqlen'].unsqueeze(dim=-1) # [B, L]
        product_id_reps = self.item_embeddings(batch['in_'+self.fiid][padding_pos]) # [N, D_item_id]

        product_reps = self.item_transform(torch.cat([product_title_reps, product_id_reps], dim=-1)) # [N, D_item]
        seq_embs = torch.split(product_reps, seqlen, dim=0) 
        seq_embs = pad_sequence(seq_embs, batch_first=True) # [B, L, D]
        
        # max_len = seq_reps.shape[1]
        # seq_item_index = torch.arange(max_len, dtype=torch.long, device=user_hist.device).unsqueeze(dim=0) # [1, L]
        # seq_item_index = seq_item_index.expand([batch_size, -1]) # [B, L] 
        # seq_item_index = seq_item_index - (max_len - seqlen.reshape(-1, 1)) # [B, L]
        # seq_item_index = seq_item_index.unsqueeze(dim=-1).expand([-1, -1, seq_reps.shape[-1]]) # [B, L, D]
        # padding_index = (seq_item_index < 0) 
        # seq_item_index[padding_index] = 0

        # seq_reps = torch.gather(seq_reps, dim=1, index=seq_item_index) # [B, L, D]
        # seq_reps = seq_reps * padding_index # set reps of padding items with all zero vector and they don't require grad.

        # get position embeddings 
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device).unsqueeze(dim=0) # [1, L]
        positions = positions.expand(user_hist.size(0), -1) # [B, L]
        padding_pos = positions >= batch['seqlen'].unsqueeze(dim=-1) # [B, L]
        positions = batch['seqlen'].unsqueeze(dim=-1) - positions # [B, L]
        positions[padding_pos] = 0
        position_embs = self.position_emb(positions)


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

        if self.training:
            if self.training_pooling_type == 'mask':
                return self.training_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token']).contiguous()
            else:
                return self.training_pooling_layer(transformer_out, batch['seqlen']).contiguous()
        else:
            if self.eval_pooling_type == 'mask':
                return self.eval_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token']).contiguous()
            else:
                return self.eval_pooling_layer(transformer_out, batch['seqlen']).contiguous()


    def encode_item(self, item_title_input):
        if item_title_input is None: # for predict
            return None 
        
        item_out = self.bert(**item_title_input, return_dict=True)
        item_hidden = item_out.last_hidden_state
        item_reps = self.sentence_embedding(item_hidden, item_title_input['attention_mask'])
        return item_reps.contiguous()
    

    def forward(self, batch : Dict):
        batch : Dict
        # session_id : [B]
        # in_product_id : [B, L]
        # seqlen : [B] 
        # pos_product_id : [B] 
        # neg_product_id : [B]
        # in_title_input : input_ids : [B, L], merge titles of last five items
        # title_input : input_ids : [B, L]
        # neg_title_input : input_ids : [B * neg, L]
        query_reps = self.encode_query(batch) # [B, D] 
        item_reps = self.encode_item(batch.get('product_title_input')) # [B, D]

        # for inference
        if query_reps is None or item_reps is None:
            return EncoderOutput(
                query_reps=query_reps,
                item_reps=item_reps,
                loss=None,
                scores=None
            )
        
        if self.training:
            pos_scores = self.scorer(query_reps, item_reps) # [B]
            neg_item_reps = self.encode_item(batch['neg_title_input']) # [B * neg, D]
            if self.negatives_x_device:
                all_neg_item_reps = self._dist_gather_tensor(neg_item_reps) # [B * neg * num_device, D]
            else:
                all_neg_item_reps = neg_item_reps # [B * neg, D]
            neg_scores = self.scorer(query_reps, all_neg_item_reps) # [B, B * neg * num_device]
            scores = torch.cat([pos_scores.unsqueeze(dim=-1), neg_scores], dim=-1) # [B, B * neg * num_device + 1]
            loss = self.loss_func(None, pos_scores, scores)
        else:
            scores = self.scorer(query_reps, item_reps)
            loss = None

        return EncoderOutput(
            scores=scores, 
            loss=loss,
            query_reps=query_reps, 
            item_reps=item_reps
        )
    

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
         

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # hf_kwargs: config and cache_dir
        # load xlm roberta
        xlm_roberta = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        sentence_pooling_method = model_args.sentence_pooling_method
        negatives_x_device = model_args.negatives_x_device
        model = cls(
            xlm_roberta=xlm_roberta, 
            sentence_pooling_method=sentence_pooling_method,
            negatives_x_device=negatives_x_device
        )
        return model
    

    @classmethod
    def load(
            cls,
            model_name_or_path,
            sentence_pooling_method,
            negatives_x_device,
            **hf_kwargs,
    ):
        # load local
        logger.info(f'load bert backbone weight from {model_name_or_path}')
        xlm_roberta = cls.TRANSFORMER_CLS.from_pretrained(
            model_name_or_path, 
            **hf_kwargs
        )
       
        model = cls(
            xlm_roberta=xlm_roberta,  
            sentence_pooling_method=sentence_pooling_method,
            negatives_x_device=negatives_x_device
        )
        return model

    def save(self, output_dir: str):
        self.xlm_roberta.save_pretrained(output_dir)
        
        
        

        


