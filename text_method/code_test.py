import datasets
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.file_utils import ModelOutput

import json 
import numpy as np 
from dataclasses import dataclass
import torch 

# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
# print(tokenizer)

with open('/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/ckpt/05221403_UK/checkpoint-54000/pytorch_model.bin', 'rb') as f:
    UK_bert = torch.load(f, 'cpu')

UK_bert


