from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
import json 


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# prepare input
text = ""
encoded_input = tokenizer(text, add_special_tokens=False, max_length=10, truncation=True)
decoded_output = tokenizer.encode_plus(encoded_input['input_ids'], max_length=10, truncation=True)
decoded_output = tokenizer.batch_decode(sequences=encoded_input['input_ids'], skip_special_tokens=True)
print(encoded_input)


# ds = load_dataset('json', data_files='./data/dev_corpus.json', split='train')
# print(ds)

# with open('./data/dev_query.json', 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line)
#     # data = json.load(f)
#     print(data)