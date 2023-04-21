import datasets
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
import json 


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# prepare input
text = "Hello, world!"
text_2 = "你好，世界！你好，世界！你好，世界！"

# text = "你好，世界！"
# text_2 = "Hello, world! Hello, world!"
encoded_input = tokenizer(text, add_special_tokens=False, max_length=10, truncation=True,
                        return_attention_mask=False, return_token_type_ids=False)
encoded_input_2 = tokenizer(text_2, add_special_tokens=False, max_length=10, truncation=True,
                        return_attention_mask=False, return_token_type_ids=False)
# decoded_output = tokenizer.encode_plus(encoded_input['input_ids'], max_length=10, truncation=True)
# decoded_output = tokenizer.batch_decode(sequences=encoded_input['input_ids'], skip_special_tokens=True)
padding_input = tokenizer.pad([encoded_input, encoded_input_2])
print(encoded_input)

# ds = datasets.Dataset.load_from_disk('./data/BertTokenizer_data/UK_corpus')
# ds = load_dataset('json', data_files='./data/UK_corpus.json', split='train')
# print(ds)

# with open('./valid_results/valid_UK_ranking.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         print(line)
        # data = json.loads(line)
    # data = json.load(f)
    # print(data)