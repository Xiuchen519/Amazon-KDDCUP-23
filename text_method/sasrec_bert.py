import sys 
sys.path = ['./RecStudio'] + sys.path
import os 
from typing import Dict, Optional
import logging 
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel, TrainingArguments
from recstudio.model.scorer import InnerProductScorer
from recstudio.model.loss_func import SoftmaxLoss
from arguments import ModelArguments
from transformers.file_utils import ModelOutput


logger = logging.getLogger(__name__)

@dataclass
class EncoderOutput(ModelOutput): # if a attribute is None, then it will not be in .items().
    query_reps: Optional[Tensor] = None
    item_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class SASRec_Bert(nn.Module):
    TRANSFORMER_CLS = AutoModel
    
    def __init__(self,
                 xlm_roberta : PreTrainedModel, 
                 sentence_pooling_method : str = 'cls', 
                 negatives_x_device : bool = False) -> None:
        super().__init__()
        self.xlm_roberta : PreTrainedModel = xlm_roberta
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
    

    def encode_query(self, query_title_input):
        # query_title_input [B, L]
        if query_title_input is None: # for predict
            return None 
        
        qry_out = self.xlm_roberta(**query_title_input, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        q_reps = self.sentence_embedding(q_hidden, query_title_input['attention_mask'])
        return q_reps.contiguous()


    def encode_item(self, item_title_input):
        if item_title_input is None: # for predict
            return None 
        
        item_out = self.xlm_roberta(**item_title_input, return_dict=True)
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
        query_reps = self.encode_query(batch.get('in_title_input')) # [B, D] 
        item_reps = self.encode_item(batch.get('title_input')) # [B, D]

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
        
        
        

        


