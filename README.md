# Amazon-KDDCUP-23
Code for Amazon KDDCUP 2023.


# Dataset 

## raw_data
copy files into raw_data or create a soft link from as below:
```
ln -s /root/autodl-tmp/xiaolong/WorkSpace/KDD_CUP_2023/kdd_cup_2023_data/* {your folder}/raw_data/
```
The config file is kdd_cup_2023.yaml

## data for debug
It is slow to load all data when debugging. You can use data_debug_sample.ipynb to sample some data for debugging.
Data config for debug data is debug_kdd_cup_2023.yaml

## KDDCUPDataset
Split sessions into train, valid, and test. The last n tokens in each session are used for trainning as autoregressive paradigm.

The model using KDDCUPSliceDataset is SASRec2

## KDDCUPSliceDataset
Split sessions into train, valid, and test. Cut each session into several slices as sliding window, and only the last item is used for trainning.

The model using KDDCUPSliceDataset is SASRec_Next
