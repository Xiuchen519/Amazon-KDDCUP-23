import datasets
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.file_utils import ModelOutput

import json 
import numpy as np 
from dataclasses import dataclass
import torch 

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
print(tokenizer)


