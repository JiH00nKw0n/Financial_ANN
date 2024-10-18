from typing import Optional, Dict

import torch
from torch.utils.data import RandomSampler
from transformers import Trainer
from transformers.trainer_utils import has_length

from transformers import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

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
        })
        outputs = model(**inputs)
        loss = outputs.loss

        accuracy = (outputs.logits.argmax(dim=1) == inputs['labels']).float().mean().detach()

        self.state.accuracy = getattr(
            self.state, "accuracy", torch.tensor(0.0).to(accuracy.device)
        )

        self.state.accuracy += accuracy

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            if hasattr(self.state, "accuracy"):
                tr_accuracy_scalar = self._nested_gather(self.state.accuracy).mean().item()
                logs["accuracy"] = round(tr_accuracy_scalar / (self.state.global_step - self._globalstep_last_logged),
                                         4)

                self.state.accuracy -= self.state.accuracy

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)