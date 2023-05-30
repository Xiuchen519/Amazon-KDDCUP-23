import sys
sys.path = ['./RecStudio'] + sys.path
import logging
import os
from pathlib import Path

import numpy as np
from trainer import KDDCupTrainer
from sasrec_bert import SASRec_Bert
from arguments_sasrec_bert import ModelArguments, DataArguments, \
    SASRecBertTrainingArguments as TrainingArguments
from data import KDDCupProductCollator, KDDCupProductCollator_Bert
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import XLMRobertaTokenizer
from recstudio.data.advance_dataset import KDDCUPSeqDataset, KDDCupCollator, KDDCUPSessionDataset
from recstudio.utils import parser_yaml

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    ) # SentencePiece is needed for XLMRoberta 
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    # Get Dataset
    data_conf = {}
    file_conf = parser_yaml(data_args.dataset_config_path)
    data_conf.update(file_conf)
    data_specified_conf = { 
        'max_title_len' : data_args.max_title_len,
        'max_desc_len' : data_args.max_desc_len,
        'split_mode' : data_args.split_mode, 
        'split_ratio' : data_args.split_ratio, 
        'tokenizer' : model_args.tokenizer_name
    }
    data_conf.update(data_specified_conf)

    datasets = KDDCUPSessionDataset.build_datasets(name=data_args.dataset_name, specific_config=data_conf)

    use_field_list = [datasets[0].fuid, datasets[0].fiid, datasets[0].frating, 'locale',
            'UK_index', 'DE_index', 'JP_index', 'ES_index', 'IT_index', 'FR_index']
    if data_args.use_session_title: use_field_list.append('session_title')
    if data_args.use_product_title: use_field_list.append('product_title')
    if data_args.use_session_desc: use_field_list.append('session_desc')
    if data_args.use_product_desc: use_field_list.append('product_desc')
    if data_args.use_session_text: use_field_list.append('session_text')
    if data_args.use_product_text: use_field_list.append('product_text')

    datasets[0].use_field = set(use_field_list)
    datasets[1].use_field = set(use_field_list)
    datasets[0].drop_feat(use_field_list)
    datasets[1].drop_feat(use_field_list)

    if training_args.do_train:
        train_dataset = datasets[0]
        train_dataset.neg_count = data_args.neg_item_num # no need to preprocess dataset again when neg count is changed.
    else:
        train_dataset = None

    # Get Model
    if training_args.do_train:
        model = SASRec_Bert.build(
            model_args,
            training_args,
            train_data=train_dataset,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    else:
        model = SASRec_Bert.load(
            model_args.model_name_or_path,
            sentence_pooling_method=model_args.sentence_pooling_method,
            negatives_x_device=model_args.negatives_x_device,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    trainer = KDDCupTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=KDDCupCollator(tokenizer),
    )


    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    # Training
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_checkpoint_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        logging.info("*** Prediction ***")
        # if os.path.exists(data_args.prediction_save_path):
        #     raise FileExistsError(f"Existing: {data_args.prediction_save_path}. Please save to other paths")

        if training_args.prediction_on == 'all_items':
            logging.info("*** All items Prediction ***")
            item_path = os.path.join(data_args.prediction_save_path, 'item_reps')
            Path(item_path).mkdir(parents=True, exist_ok=True)

            test_dataset = datasets[0].title_feat # 0 is the padding token 
            trainer.data_collator = KDDCupProductCollator_Bert(tokenizer, 'session_title_input')

            item_reps = trainer.predict(test_dataset=test_dataset).predictions

            if trainer.is_world_process_zero():
                assert len(test_dataset) == len(item_reps)
                np.save(os.path.join(item_path, 'item.npy'), item_reps[1:]) # exclude the first item. 
        else:

            if training_args.prediction_on == 'valid':
                logging.info("*** Validation dataset Prediction ***")
                test_dataset = datasets[1]
                test_dataset.use_field = use_field_list
                test_dataset.predict_mode = True # set mode to prediction.
                test_dataset.eval_mode = True 
                query_path = os.path.join(data_args.prediction_save_path, 'valid_query_reps')
                Path(query_path).mkdir(parents=True, exist_ok=True)
                
            elif training_args.prediction_on == 'test':
                logging.info("*** Test dataset Prediction ***")
                test_dataset = datasets[0].build_test_dataset(data_args.prediction_data_path)
                test_dataset.use_field = use_field_list
                test_dataset.drop_feat(use_field_list)
                test_dataset.predict_mode = True # set mode to prediction.
                test_dataset.eval_mode = False 
                query_path = os.path.join(data_args.prediction_save_path, 'test_query_reps')
                Path(query_path).mkdir(parents=True, exist_ok=True)
                

            else: 
                raise ValueError(f'{training_args.prediction_on} is not valid for prediction on.')
            
            query_reps = trainer.predict(test_dataset=test_dataset).predictions

            if trainer.is_world_process_zero():
                assert len(test_dataset) == len(query_reps)
                np.save(os.path.join(query_path, 'query.npy'), query_reps)
            

if __name__ == "__main__":
    main()