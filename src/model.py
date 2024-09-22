from typing import Optional, Union, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import TokenClassifierOutput

from .registry import registry

MODEL_TYPE = 'base'


@registry.register_model_config("BaseConfig")
class BaseConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
            self,
            hidden_size: int = 4096,
            intermediate_size: int = 8192,
            hidden_act: str = 'relu',
            classifier_dropout: float = 0.0,
            hidden_droput: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.classifier_dropout = classifier_dropout
        self.hidden_dropout = hidden_droput


class MLPWithTwoHiddenLayers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)

        self.fc2 = nn.Linear(config.intermediate_size, config.intermediate_size)

        self.fc3 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, input_embeds: Tensor) -> Tensor:
        hidden_states = self.fc1(input_embeds)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        output = self.fc3(hidden_states)
        return output


class BasePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BaseConfig
    base_model_prefix = "base"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_model("BaseModelForTokenClassification")
class BaseModelForTokenClassification(BasePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MLPWithTwoHiddenLayers(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        sequence_output = self.model(
            inputs_embeds=inputs_embeds,
        )
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )


AutoConfig.register(MODEL_TYPE, BaseConfig)
AutoModel.register(BaseConfig, BaseModelForTokenClassification)
