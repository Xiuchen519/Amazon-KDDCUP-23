# Package
1. install sentencepiece
   ```
   pip install sentencepiece
   ```

# Generate data
1. place raw_data in ./raw_data. Test data in phase1 is renamed by 'session_test_taskx_phase1.csv' and test data in phase2 is named by 'sessions_test_taskx.csv'

2. run ./data_preprocess/product_feature_process.ipynb to process product data 
It will create ./data_for_recstudio, and generate processed_products_train.csv in ./data_for_recstudio. 
Move processed_products_train.csv to ./data_for_recstudio/task1_data 

3. run ./data_preprocess/phase2/data_split.ipynb to generate train and valid data for task1. About 5 mins.


# Run xlm-RoBertA
1. specify GPUs in ```CUDA_VISIBLE_DEVICES``` in ./saved_scripts/run_xlm_roberta.bash 
2. run ./saved_scripts/run_xlm_roberta.bash
3. data processing takes about 30 mins. 