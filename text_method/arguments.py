import os
from dataclasses import dataclass, field
from typing import Optional, Union
from transformers import TrainingArguments



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    sentence_pooling_method: str = field(default='cls')

    negatives_x_device : bool = field(default=False, metadata={"help" : "whether to use all negative items in all gpus."})



@dataclass
class DataArguments:

    dataset_config_path: str = field(
        default='/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/RecStudio/recstudio/data/config/kdd_cup_2023_UK_bert.yaml',
        metadata={"help" : "Path of RecStudio Dataset config file."}
    )

    prediction_data_path: Optional[str] = field(
        default='/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/data_for_recstudio/test_inter_feat.csv',  
        metadata={"help": "Path of prediction data."}
    )

    prediction_save_path: Optional[str] = field(
        default=None, metadata={"help": "Path to save prediction."}
    )

    max_seq_len: int = field(
        default=5, 
        metadata={"help" : "The maximum products in a session slice."}
    )

    neg_item_num: int = field(
        default=8, 
        metadata={"help" : "Number of negative items for one postive item."}
    )

    use_session_title: bool = field(default=False, metadata={"help" : "whether to use session title."})

    use_product_title: bool = field(default=False, metadata={"help" : "whether to use product title."})

    use_session_desc: bool = field(default=False, metadata={"help" : "whether to use session desc."})

    use_product_desc: bool = field(default=False, metadata={"help" : "whether to use product desc."})


    max_title_len: int = field(
        default=70, 
        metadata={
            "help": "The maximum total input sequence length after tokenization for product title. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )

    max_desc_len: int = field(
        default=70, 
        metadata={
            "help": "The maximum total input sequence length after tokenization for product desc. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )

    split_mode: str = field(
        default='user',
        metadata={
            "help": "The split mode of dataset."
        },
    )

    split_ratio: list[float] = field(
        default_factory=lambda:[1.0],
        metadata={
            "help": "The split ratio of dataset."
        },
    )


@dataclass
class SASRecBertTrainingArguments(TrainingArguments):
    prediction_on : str = field(default='valid', metadata={"help" : "predict on valid dataset, test dataset or all items."})
    