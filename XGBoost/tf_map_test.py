from datasets import Dataset as TFDataset 

def get_sess_scores(sess):
    sess_id = sess['sess_id']
    return {'sess_bm25_scores' : 1.0}


def get_batch_sess_scores(sess):
    sess_id_list = sess['sess_id']
    return {'sess_bm25_scores' : [1.0 for _ in sess_id_list]}


test_query_dataset = TFDataset.from_dict({'sess_id' : list(range(100000))})
# test_query_dataset = test_query_dataset.map(get_sess_scores, num_proc=10, batched=False)
test_query_dataset = test_query_dataset.map(get_batch_sess_scores, num_proc=10, batch_size=1, batched=True)

print('')