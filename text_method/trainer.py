import sys
sys.path = ['./RecStudio'] + sys.path
from transformers.trainer import * 
from recstudio.data.advance_dataset import KDDCUPSeqDataset, KDDCUPSessionDataset
from torch.cuda.amp import autocast

class KDDCupTrainer(Trainer):
    
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


    def get_train_dataloader(self) -> DataLoader:
        train_sampler = self._get_train_sampler()

        self.train_dataset : KDDCUPSeqDataset
        self.train_dataset.use_field = set(
            [self.train_dataset.fuid, self.train_dataset.fiid, self.train_dataset.frating, 'locale', 'title',
             'UK_index', 'DE_index', 'JP_index', 'ES_index', 'IT_index', 'FR_index']
        )
        self.train_dataset.predict_mode = False 
        self.train_dataset.eval_mode = False 
        return DataLoader(self.train_dataset, 
                          batch_size=self.args.train_batch_size,
                          sampler=train_sampler,
                          num_workers=self.args.dataloader_num_workers,
                          drop_last=True, # ensure batch size is same on different gpus.
                          collate_fn=self.data_collator,
                          pin_memory=self.args.dataloader_pin_memory
                        )
    

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        test_sampler = self._get_eval_sampler(test_dataset)
        # We use the same batch_size as for eval.
        if isinstance(test_dataset, KDDCUPSessionDataset):
            test_dataset.use_field = set(
                [test_dataset.fuid, test_dataset.fiid, test_dataset.frating, 'locale', 'title',
                'UK_index', 'DE_index', 'JP_index', 'ES_index', 'IT_index', 'FR_index']
            )
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(inputs) # don't unpack inputs
        loss = outputs.loss
        if torch.isnan(loss):
            print('cxl')
            raise RuntimeError()

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self,
                        model: nn.Module,
                        inputs: Dict[str, Union[torch.Tensor, Any]], 
                        prediction_loss_only: bool, 
                        ignore_keys: Optional[List[str]] = None
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(inputs) # don't unpack 
            else:
                outputs = model(inputs) # don't unpack 

            loss = None
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None
        return (loss, logits, labels)

    
    
    
