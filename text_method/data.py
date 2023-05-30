import torch
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding

class KDDCupProductCollator :

    def __init__(self, tokenizer) -> None:
        self.tokenizer : PreTrainedTokenizer = tokenizer

    def __call__(self, batch):

        collated_batch = {}
        for feat_key in batch[0].keys():
            feat_list = []
            for i in range(len(batch)):
                feat_list.append(batch[i][feat_key])
            
            if feat_key == 'input_ids': # list[list]
                collated_batch['title_input'] = self.tokenizer.pad(
                    {'input_ids' : feat_list}, 
                    padding=True, 
                    return_tensors='pt'
                )
            else:
                collated_batch[feat_key] = torch.tensor(feat_list)
            
        return collated_batch
    

class KDDCupProductCollator_Bert :

    def __init__(self, tokenizer, product_feature_name) -> None:
        self.tokenizer : PreTrainedTokenizer = tokenizer
        self.product_feature_name = product_feature_name

    def __call__(self, batch):

        collated_batch = {}
        for feat_key in batch[0].keys():
            feat_list = []
            for i in range(len(batch)):
                feat_list.append(batch[i][feat_key])
            
            if feat_key == 'input_ids': # list[list]
                collated_batch[self.product_feature_name] = self.tokenizer.pad(
                    {'input_ids' : feat_list}, 
                    padding=True, 
                    return_tensors='pt'
                )
            else:
                collated_batch[feat_key] = torch.tensor(feat_list)
            
        return collated_batch

     
