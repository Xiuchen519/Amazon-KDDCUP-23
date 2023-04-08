import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from typing import List

import os
import numpy as np
import pandas as pd
from functools import lru_cache
import json
from tqdm import tqdm 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_title_length", type=int, default=512)
    return parser.parse_args()
 

def tokenize_function(examples):
    if 'title' in examples:
        return tokenizer(examples['title'], add_special_tokens=False, truncation=True, max_length=max_title_length,
                         return_attention_mask=False,
                         return_token_type_ids=False)
    elif 'title_list' in examples:
        batch_size = len(examples['id'])
        title_lists = examples['title_list']
        title_lists_len = list(map(lambda x : len(x), title_lists)) # [B]
        flattened_titles = np.concatenate(title_lists, axis=0).tolist() # [N]
        flattened_encoded_titles:list[list] = tokenizer(flattened_titles, add_special_tokens=False, truncation=True, max_length=max_title_length, 
                                  return_attention_mask=False,
                                  return_token_type_ids=False)
        flattened_encoded_titles = tokenizer.batch_decode(sequences=flattened_encoded_titles['input_ids'], skip_special_tokens=True)

        content = []
        title_lists_len = np.array(title_lists_len).cumsum().tolist()
        title_lists_len = [0] + title_lists_len
        for i in range(batch_size):
            start = title_lists_len[i]
            end = title_lists_len[i + 1]
            title_list_str = flattened_encoded_titles[start]
            for j in range(start + 1, end):
                title_list_str = title_list_str + tokenizer.sep_token + flattened_encoded_titles[j]
            content.append(title_list_str)

        encoded_input = tokenizer(content, add_special_tokens=False, truncation=True, max_length=max_title_length * 5 + 10,
                         return_attention_mask=False,
                         return_token_type_ids=False)
        return encoded_input

# functions to save data as json 

def read_product_data():
    return pd.read_csv(os.path.join(raw_data_dir, f'products_train.csv'))

def read_train_data():
    return pd.read_csv(os.path.join(data_for_recstudio, 'all_task_1_train_sessions.csv'))

def read_valid_data():
    return pd.read_csv(os.path.join(data_for_recstudio, 'all_task_1_valid_sessions.csv'))

def read_test_data(task):
    return pd.read_csv(os.path.join(raw_data_dir, f'sessions_test_{task}.csv'))

def read_test1_data(locale):
    return pd.read_csv(os.path.join(data_for_recstudio, f'session_test_task1_{locale}.csv'))

# save as json 
# id_locale as new index
def save_corpus_as_json(corpus_data, corpus_path, map_id_path):
    with open(corpus_path, 'w', encoding='utf-8') as f, open(map_id_path, 'w', encoding='utf-8') as fid:
        for i in tqdm(range(len(corpus_data))):
            product = corpus_data.iloc[i]
            if pd.isna(product['title']):
                data = {'id' : f"{product['id']}_{product['locale']}", 'title' : " "}
            else:
                data = {'id' : f"{product['id']}_{product['locale']}", 'title' : product['title']}
            f.write(json.dumps(data) + '\n')
            fid.write(f"{product['id']}_{product['locale']}" + "\t" + str(i) + "\n")

def save_dev_corpus_as_json(corpus_path, map_id_path):
    with open(corpus_path, 'w', encoding='utf-8') as f, open(map_id_path, 'w', encoding='utf-8') as fid:
        for i in tqdm(range(500000, 1000000)):
            product = product_data.iloc[i]
            if pd.isna(product['title']):
                data = {'id' : f"{product['id']}_{product['locale']}", 'title' : " "}
            else:
                data = {'id' : f"{product['id']}_{product['locale']}", 'title' : product['title']}
            f.write(json.dumps(data) + '\n')
            fid.write(f"{product['id']}_{product['locale']}" + "\t" + str(i) + "\n")

# save query as json
def save_query_as_json(sessions, query_path, map_id_path, qrels_path=None, sess_len=5, test=False):
    if not test: 
        with open(query_path, 'w', encoding='utf-8') as f, open(map_id_path, 'w', encoding='utf-8') as fid, \
            open(qrels_path, 'w', encoding='utf-8') as f_qrel:
            for i in tqdm(range(len(sessions))):
                sess = sessions.iloc[i]
                sess_locale = sess['locale']
                prev_items = sess['prev_items']
                next_item = sess['next_item']

                prev_items = eval(prev_items.replace(" ", ","))
                prev_items = prev_items[-sess_len:]
                prev_items = list(map(lambda x : x + "_"+ sess_locale, prev_items))
                # prev_items = list(map(lambda x : product_map_id[x], prev_items))
                prev_items_titles = reindex_product_data.loc[prev_items]['title'].to_list()

                # train query
                data = {'id' : str(i), 'title_list' : prev_items_titles}
                f.write(json.dumps(data) + '\n')
                fid.write(str(i) + "\t" + str(i) + "\n")

                # train qrel
                f_qrel.write(str(i) + "\t" + next_item + '_' + sess_locale + "\n")
    else:
        with open(query_path, 'w', encoding='utf-8') as f, open(map_id_path, 'w', encoding='utf-8') as fid:
            for i in tqdm(range(len(sessions))):
                sess = sessions.iloc[i]
                sess_locale = sess['locale']
                prev_items = sess['prev_items']

                prev_items = eval(prev_items.replace(" ", ","))
                prev_items = prev_items[-sess_len:]
                prev_items = list(map(lambda x : x + "_"+ sess_locale, prev_items))
                # prev_items = list(map(lambda x : product_map_id[x], prev_items))
                prev_items_titles = reindex_product_data.loc[prev_items]['title'].to_list()

                # train query
                data = {'id' : str(i), 'title_list' : prev_items_titles}
                f.write(json.dumps(data) + '\n')
                fid.write(str(i) + "\t" + str(i) + "\n")


def save_dev_query_as_json(train_query_path, map_id_path, train_qrels_path, sess_len=5):
    with open(train_query_path, 'w', encoding='utf-8') as f, open(map_id_path, 'w', encoding='utf-8') as fid, \
        open(train_qrels_path, 'w', encoding='utf-8') as f_qrel:
        for i in tqdm(range(50000)):
            sess = train_sessions.iloc[i]
            sess_locale = sess['locale']
            prev_items = sess['prev_items']
            next_item = sess['next_item']
            prev_items = eval(prev_items.replace(" ", ","))
            prev_items = prev_items[-sess_len:]
            prev_items = list(map(lambda x : x + "_"+ sess_locale, prev_items))
            prev_items_titles = reindex_product_data.loc[prev_items]['title'].to_list()

            # train query
            data = {'id' : str(i), 'title_list' : prev_items_titles}
            f.write(json.dumps(data) + '\n')
            fid.write(str(i) + "\t" + str(i) + "\n")

            # train qrel
            f_qrel.write(str(i) + "\t" + next_item + '_' + sess_locale + "\n")

args = get_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
max_title_length = args.max_title_length

data_for_recstudio  = '/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio'
raw_data_dir = '/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/raw_data'

if __name__ == '__main__':
    print(os.getcwd())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'corpus')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'UK_corpus')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'dev_corpus')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'train_query')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'dev_query')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'train_UK_query')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'valid_UK_query')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'test1_UK_query')).mkdir(parents=True, exist_ok=True)


    '''preprocess'''
    product_data = read_product_data()
    train_sessions = read_train_data()
    valid_sessions = read_valid_data()
    test1_UK_sessions = read_test1_data('UK')
    # title fill nan 
    product_data['title'] = product_data['title'].fillna(tokenizer.unk_token)
    reindex_product_data = product_data.set_index(product_data['id'] + '_' + product_data['locale'])

    # filter UK sessions 
    UK_product_data = product_data.query('locale == "UK"').reset_index(drop=True)
    train_UK_sessions = train_sessions.query('locale == "UK"').reset_index(drop=True)
    valid_UK_sessions = valid_sessions.query('locale == "UK"').reset_index(drop=True)
    # dev sessions 
    dev_sessions = train_sessions.sample(50000).reset_index(drop=True)

    # save_corpus_as_json(product_data, './data/corpus.json', os.path.join(args.output_dir, 'corpus/mapping_id.txt'))

    save_corpus_as_json(UK_product_data, './data/UK_corpus.json', os.path.join(args.output_dir, 'UK_corpus/mapping_id.txt'))

    # save_query_as_json(train_sessions, './data/train_query.json', os.path.join(args.output_dir, 'train_query/mapping_id.txt'), 
    #                    os.path.join(args.output_dir, 'train_qrels.txt'))

    save_query_as_json(train_UK_sessions, './data/train_UK_query.json', os.path.join(args.output_dir, 'train_UK_query/mapping_id.txt'),  
                       os.path.join(args.output_dir, 'train_UK_qrels.txt'))

    save_query_as_json(valid_UK_sessions, './data/valid_UK_query.json', os.path.join(args.output_dir, 'valid_UK_query/mapping_id.txt'),  
                       os.path.join(args.output_dir, 'valid_UK_qrels.txt'))

    save_query_as_json(test1_UK_sessions, './data/test1_UK_query.json', os.path.join(args.output_dir, 'test1_UK_query/mapping_id.txt'),
                       test=True)

    save_query_as_json(dev_sessions, './data/dev_query.json', os.path.join(args.output_dir, 'dev_query/mapping_id.txt'),  
                       os.path.join(args.output_dir, 'dev_qrels.txt'))

    


    
    '''save dataset'''
    corpus = load_dataset('json', data_files='./data/corpus.json', split='train')
    corpus = corpus.map(tokenize_function, num_proc=8, remove_columns=["title"], batched=True)
    corpus.save_to_disk(os.path.join(args.output_dir, 'corpus'))
    print('corpus dataset:', corpus)

    UK_corpus = load_dataset('json', data_files='./data/UK_corpus.json', split='train')
    UK_corpus = UK_corpus.map(tokenize_function, num_proc=8, remove_columns=["title"], batched=True)
    UK_corpus.save_to_disk(os.path.join(args.output_dir, 'UK_corpus'))
    print('UK corpus dataset:', UK_corpus)

    # train_query = load_dataset('json', data_files='./data/train_query.json', split='train')
    # train_query = train_query.map(tokenize_function, num_proc=8, remove_columns=["title_list"], batched=True)
    # train_query.save_to_disk(os.path.join(args.output_dir, 'train_query'))
    # print('train query dataset:', train_query)

    train_UK_query = load_dataset('json', data_files='./data/train_UK_query.json', split='train')
    train_UK_query = train_UK_query.map(tokenize_function, num_proc=8, remove_columns=["title_list"], batched=True)
    train_UK_query.save_to_disk(os.path.join(args.output_dir, 'train_UK_query'))
    print('train UK query dataset:', train_UK_query)

    valid_UK_query = load_dataset('json', data_files='./data/valid_UK_query.json', split='train')
    valid_UK_query = valid_UK_query.map(tokenize_function, num_proc=8, remove_columns=["title_list"], batched=True)
    valid_UK_query.save_to_disk(os.path.join(args.output_dir, 'valid_UK_query'))
    print('valid UK query dataset:', valid_UK_query)

    test1_UK_query = load_dataset('json', data_files='./data/test1_UK_query.json', split='train')
    test1_UK_query = test1_UK_query.map(tokenize_function, num_proc=8, remove_columns=["title_list"], batched=True)
    test1_UK_query.save_to_disk(os.path.join(args.output_dir, 'test1_UK_query'))
    print('test1 UK query dataset:', test1_UK_query)

    dev_query = load_dataset('json', data_files='./data/dev_query.json', split='train')
    dev_query = dev_query.map(tokenize_function, num_proc=8, remove_columns=["title_list"], batched=True)
    dev_query.save_to_disk(os.path.join(args.output_dir, 'dev_query'))
    print('dev query dataset:', dev_query)
