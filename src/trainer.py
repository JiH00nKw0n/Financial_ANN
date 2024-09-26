import torch
from transformers import Trainer
from torch.utils.data import SequentialSampler
import datasets
from typing import Optional

from transformers.trainer_pt_utils import LengthGroupedSampler

from .registry import registry

@registry.register_trainer('SequentialTrainer')
class SequentialTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Returns a SequentialSampler for the training dataset instead of the default RandomSampler.
        """
        if self.train_dataset is None or not hasattr(self.train_dataset, "__len__"):
            return None

        # Build the sampler. If group_by_length is set, use LengthGroupedSampler, otherwise use SequentialSampler.
        if self.args.group_by_length:
            if datasets and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        # Use SequentialSampler for ordered iteration.
        return SequentialSampler(self.train_dataset)