import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Union, Optional, List, Tuple
from transformers import Trainer
from transformers.utils import cached_property, is_torch_npu_available
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

class CHATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _save(self, output_dir: str, state = None, **kwargs):
        # TODO
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        last_checkpoint = get_last_checkpoint(os.path.dirname(output_dir) if "checkpoint" in output_dir else output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if last_checkpoint is not None:
            logger.info(f"Deleting previous checkpoint: {last_checkpoint}")
            shutil.rmtree(last_checkpoint)

        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    # def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
    #     # TODO
    #     model = self.model.lm_model
    #     model = super()._load_from_checkpoint(resume_from_checkpoint, model)

    #     with torch.no_grad():
    #         if 'embedding' in ckpt:
    #             self.model.MTP.copy_(ckpt["embedding"])
    #             self.model.MTP.required_grad = True
    #         # if 'lm_head' in ckpt:
    #         #     self.model.lm_model.get_output_embeddings().weight.data[mtp_id] = ckpt["lm_head"].to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model = self.unwrap_model(model)
        if is_torch_npu_available():
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = inputs['position_ids']
        label_ids = inputs['label_ids']
        is_beacon = inputs['is_beacon'].to(torch.bool)
        
        label_ids = label_ids[0]
        logits = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            is_beacon = is_beacon
        ).logits[0][-len(label_ids)-1:-1]

        next_token_logits = logits
        next_token_ids = torch.argmax(next_token_logits, dim=-1)
        print(f"Next token ids: {next_token_ids}")
        print(f"Label ids: {label_ids}")
        mask = label_ids != -100
        with torch.no_grad():
            acc = torch.mean((next_token_ids == label_ids).float()[mask]).item()
        print(f"Accuracy: {acc:.4f}")
        # if self.is_world_process_zero():
        #     self.log({"accuracy": acc})
        print(f"Logits shape: {logits.shape}, Label ids shape: {label_ids.shape}")
        loss = F.cross_entropy(logits, label_ids)

        outputs = {"loss": loss, "logits": logits, 'labels': label_ids}

        return (loss, outputs) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        logits = tuple(v for k, v in outputs.items() if k not in ["loss", "labels"])
        labels = outputs["labels"]
        # print(logits)
        return (loss, logits, labels)

    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        return batch_samples, num_items_in_batch

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        return 0