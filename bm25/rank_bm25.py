#!/usr/bin/env python

import math
import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer
from multiprocessing import Pool, cpu_count
from datasets import Dataset as TFDataset 
from functools import partial
import pandas as pd 
import logging 
from tqdm import tqdm 
import pickle
import numba
from numba import jit

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, corpus_col_name, tokenizer, max_length=200):
        self.logger = logging.getLogger('bm25_logger')
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus, corpus_col_name, max_length)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

        self.doc_len = np.array(self.doc_len)

    def _initialize(self, corpus):
        self.logger.info('********* start to initialize BM25 ************')
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in tqdm(corpus):
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        self.logger.info('********* BM25 is initialized ************')
        return nd


    def _tokenize_corpus(self, corpus : pd.DataFrame, corpus_col_name='title', max_length=200) -> list:
        self.logger.info('********* start to tokenizer corpus ************')
        
        def tokenize_function(examples, tokenizer, max_length):
            sentence = examples[corpus_col_name]
            if corpus_col_name in examples:
                return tokenizer(examples[corpus_col_name], 
                    add_special_tokens=False, # don't add special tokens when preprocess
                    truncation=True, 
                    max_length=max_length,
                    return_attention_mask=False,
                    return_token_type_ids=False)
            
        corpus = TFDataset.from_pandas(corpus, preserve_index=False)
        corpus = corpus.map(partial(tokenize_function, tokenizer=tokenizer, max_length=max_length), 
                                        num_proc=8, remove_columns=[corpus_col_name], batched=True)
        self.logger.info('********* corpus is tokenized ************')
        
        return corpus['input_ids']


    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, corpus_col_name:str, tokenizer=None, max_length=200, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, corpus_col_name=corpus_col_name, tokenizer=tokenizer, max_length=max_length)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        self.logger.info('********* start to calculate idf ************')
        negative_idfs = []
        for word, freq in tqdm(nd.items(), total=len(nd)):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps
        self.logger.info('********* idf is calculated ************')
    

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = self.doc_len
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = self.doc_len[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()
    

if __name__ == '__main__':
    TOKENIZER_NAME = 'xlm-roberta-base'
    PRODUCT_DATA_PATH = '/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/raw_data/products_train.csv'
    MAX_TITLE_LEN = 200
    MAX_DESC_LEN = 500
    TITLE_SAVE_PATH = '/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/bm25/cache/title_bm25.pkl'
    DESC_SAVE_PATH = '/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/bm25/cache/desc_bm25.pkl'

    logger = logging.getLogger('bm25_logger')
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('[%(asctime)s]-[%(levelname)s]: %(message)s'))
    logger.addHandler(sh)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=False,
    )

    product_data = pd.read_csv(PRODUCT_DATA_PATH)


    # title_corpus = product_data[['title']]
    # title_corpus = pd.concat([pd.DataFrame({'title' : ['']}), title_corpus]).reset_index(drop=True) # add padding product
    # title_corpus['title'] = title_corpus['title'].fillna('')

    # title_bm25 = BM25Okapi(title_corpus, corpus_col_name='title', tokenizer=tokenizer, max_length=MAX_TITLE_LEN, k1=1.5, b=0.75, epsilon=0.25)

    # with open(TITLE_SAVE_PATH, 'wb') as f:
    #     pickle.dump(title_bm25, f)
    #     logger.info(f'title BM25 is saved in {TITLE_SAVE_PATH}')


    desc_corpus = product_data[['desc', 'brand', 'color', 'size', 'model', 'material', 'author']]
    padding_df = pd.DataFrame({'desc' : [''], 'brand' : [''], 'color' : [''], 'size' : [''], 'model' : [''], 'material' : [''], 'author' : ['']})
    desc_corpus = pd.concat([padding_df, desc_corpus]).reset_index(drop=True) # add padding product
    desc_corpus['desc'] = desc_corpus['desc'].fillna('')
    desc_corpus['brand'] = desc_corpus['brand'].fillna('')
    desc_corpus['color'] = desc_corpus['color'].fillna('')
    desc_corpus['size'] = desc_corpus['size'].fillna('')
    desc_corpus['model'] = desc_corpus['model'].fillna('')
    desc_corpus['material'] = desc_corpus['material'].fillna('')
    desc_corpus['author'] = desc_corpus['author'].fillna('')

    desc_corpus['desc'] = desc_corpus['desc'] + ' ' + desc_corpus['brand'] + ' ' + desc_corpus['color'] + ' ' + desc_corpus['size'] + ' ' + desc_corpus['model'] \
        + ' ' + desc_corpus['material'] + ' ' + desc_corpus['author']
    desc_corpus['desc'] = desc_corpus['desc'].apply(lambda x : x.lower())

    desc_bm25 = BM25Okapi(desc_corpus, corpus_col_name='desc', tokenizer=tokenizer, max_length=MAX_DESC_LEN, k1=1.5, b=0.75, epsilon=0.25)

    with open(DESC_SAVE_PATH, 'wb') as f:
        pickle.dump(desc_bm25, f)
        logger.info(f'desc BM25 is saved in {DESC_SAVE_PATH}')

    # with open('/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/bm25/cache/title_bm25.pkl', 'rb') as f:
    #     title_BM25 = pickle.load(f)

    # title_BM25.get_batch_scores([13, 19943, 491, 189932, 7, 165, 139379, 18266, 18266, 18266, 18266, 18266], [0, 1, 2, 3])

    

