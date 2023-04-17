from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
import json 


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# prepare input
# text = ["neural network based methods usually have better performance for session-based recommendation", 
#         "We compare our method with classic methods as well as state-ofthe-art models.", 
#         "The following nine baseline models are evaluated"]
# encoded_input = tokenizer(text, add_special_tokens=False, max_length=10, truncation=True)
# # decoded_output = tokenizer.encode_plus(encoded_input['input_ids'], max_length=10, truncation=True)
# decoded_output = tokenizer.batch_decode(sequences=encoded_input['input_ids'], skip_special_tokens=True)
# print(encoded_input)


ds = load_dataset('json', data_files='./data/UK_corpus.json', split='train')
print(ds)

# with open('./data/BertTokenizer_data/UK_corpus/mapping_id.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         print(line)
#     # data = json.load(f)
#     print(line)