import datasets
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.file_utils import ModelOutput

import json 
import numpy as np 
from dataclasses import dataclass
import torch 

# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=False)
# # model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
# print(tokenizer)
# # prepare input
# text = "Hello, world!"
# text_2 = "你好，世界！你好，世界！你好，世界！"

# # text = "你好，世界！"
# # text_2 = "Hello, world! Hello, world!"
# encoded_input = tokenizer.encode(text, add_special_tokens=False, max_length=10, truncation=True,
#                         return_attention_mask=False, return_token_type_ids=False)
# encoded_input_2 = tokenizer(text_2, add_special_tokens=False, max_length=10, truncation=True,
#                         return_attention_mask=False, return_token_type_ids=False)
# # decoded_output = tokenizer.encode_plus(encoded_input['input_ids'], max_length=10, truncation=True)
# # decoded_output = tokenizer.batch_decode(sequences=encoded_input['input_ids'], skip_special_tokens=True)
# padding_input = tokenizer.pad([encoded_input, encoded_input_2])
# print(encoded_input)

# # ds = datasets.Dataset.load_from_disk('./data/BertTokenizer_data/UK_corpus')
# # ds = load_dataset('json', data_files='./data/UK_corpus.json', split='train')
# # print(ds)

# # with open('./valid_results/valid_UK_ranking.txt', 'r', encoding='utf-8') as f:
# #     for line in f:
# #         print(line)
#         # data = json.loads(line)
#     # data = json.load(f)
    # print(data)

# q_reps = np.load('/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/text_method/results_epoch_4/item_reps/item.npy')
# print(q_reps.shape)




# @dataclass
# class EncoderOutput(ModelOutput):
#     q_reps: int = None
#     p_reps: int = None
#     loss: int = None
#     scores: int = None

# test_output = EncoderOutput(
#     q_reps=12, 
#     p_reps=None, 
#     loss=None,
#     scores=None
# )

# print(test_output)

# train_config = torch.load('/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RetroMAE/examples/retriever/kdd_cup/ckpt/04051843/training_args.bin')
train_config = torch.load('/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RetroMAE/examples/retriever/kdd_cup/ckpt/04051843/training_args.bin')

print(train_config)