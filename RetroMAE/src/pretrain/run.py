import logging
import os
import sys

import transformers
from pretrain.arguments import DataTrainingArguments, ModelArguments
from pretrain.data import DatasetForPretraining, RetroMAECollator
from pretrain.modeling import RetroMAEForPretraining
from pretrain.trainer import PreTrainer
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    AutoConfig,
    HfArgumentParser, set_seed, )
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)


class TrainerCallbackForSaving(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.should_save = True


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    training_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    if model_args.model_name_or_path:
        model = RetroMAEForPretraining.from_pretrained(model_args, model_args.model_name_or_path)
        logger.info(f"------Load model from {model_args.model_name_or_path}------")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
        bert = BertForMaskedLM(config)
        model = RetroMAEForPretraining(bert, model_args)
        logger.info("------Init the model------")
        tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    else:
        raise ValueError("You must provide the model_name_or_path or config_name")

    dataset = DatasetForPretraining(data_args.data_dir)
    data_collator = RetroMAECollator(tokenizer,
                                     encoder_mlm_probability=data_args.encoder_mlm_probability,
                                     decoder_mlm_probability=data_args.decoder_mlm_probability,
                                     max_seq_length=data_args.max_seq_length)

    # Initialize our Trainer
    trainer = PreTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.add_callback(TrainerCallbackForSaving())

    # # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload


if __name__ == "__main__":
    main()
