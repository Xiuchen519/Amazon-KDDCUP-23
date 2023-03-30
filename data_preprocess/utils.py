import os
import pandas as pd
from functools import lru_cache

train_data_dir = '../raw_data/'
test_data_dir = '../raw_data/'
task = 'task1'
PREDS_PER_SESSION = 100

# Cache loading of data for multiple calls

@lru_cache(maxsize=1)
def read_product_data():
    print(os.getcwd())
    print(os.path.join(train_data_dir, 'products_train.csv'))
    return pd.read_csv(os.path.join(train_data_dir, 'products_train.csv'))

@lru_cache(maxsize=1)
def read_train_data():
    return pd.read_csv(os.path.join(train_data_dir, 'sessions_train.csv'))

@lru_cache(maxsize=3)
def read_test_data(task):
    return pd.read_csv(os.path.join(test_data_dir, f'sessions_test_{task}.csv'))

__all__ = ["read_product_data", "read_train_data", "read_test_data"]