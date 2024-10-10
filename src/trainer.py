from typing import Optional

import torch
from torch.utils.data import RandomSampler
from transformers import Trainer
from transformers.trainer_utils import has_length

from .registry import registry

@registry.register_trainer('RandomSamplerTrainer')
class RandomSamplerTrainer(Trainer):
    """
    A subclass of the `BaseTrainer` that overrides the method for
    getting the sampler used for the training dataset. The sampler used
    in this class is a basic `RandomSampler`.
    """

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Returns a `RandomSampler` for the training dataset.

        Returns:
            `Optional[torch.utils.data.Sampler]`: A `RandomSampler` for the dataset if available, otherwise `None`.

        Raises:
            ValueError: If `group_by_length` is set to `True`, since this trainer doesn't support grouping by length.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            raise ValueError("Argument `group_by_length` must be `False`.")
        else:
            return RandomSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for the current batch of inputs using the negative CLIP loss function.

        Args:
            model: The model to be used for generating the logits.
            inputs: A dictionary of inputs that includes features such as images and captions.
            return_outputs (`bool`, *optional*, defaults to `False`): If `True`, returns both the loss and the model outputs.

        Returns:
            `torch.Tensor` or `Tuple[torch.Tensor, Any]`: If `return_outputs` is `True`, returns a tuple of (loss, outputs).
            Otherwise, returns only the loss.
        """
        inputs = dict(inputs, **{
            'return_dict': True,
            'return_loss': True,
        })
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss