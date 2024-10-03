from typing import Optional, Union, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoModel, AutoConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import TokenClassifierOutput

from .registry import registry
from .base import BaseModelConfig, BasePreTrainedModel

CEL_MODEL_TYPE = 'mlp'


@registry.register_model_config("MLPModelConfig")
class MLPModelConfig(BaseModelConfig):
    model_type = CEL_MODEL_TYPE



class MLPWithTwoHiddenLayers(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.activation_fn = ACT2FN[config.hidden_act]

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)

        self.fc2 = nn.Linear(config.intermediate_size, config.intermediate_size)

        self.fc3 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        output = self.fc3(hidden_states)
        return output


@registry.register_model("MLPModelForTokenClassification")
class MLPModelForTokenClassification(BasePreTrainedModel):
    config_class = MLPModelConfig
    base_model_prefix = CEL_MODEL_TYPE
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layers = nn.ModuleList(
            [MLPWithTwoHiddenLayers(config, layer_idx) for layer_idx in range(config.num_mlp_layers)]
        )
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

        hidden_states = inputs_embeds

        for mlp_layer in self.layers:
            layer_output = mlp_layer(
                hidden_states=hidden_states,
            )
            layer_output = self.dropout(layer_output)

        logits = self.score(layer_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )


AutoConfig.register(CEL_MODEL_TYPE, MLPModelConfig)
AutoModel.register(MLPModelConfig, MLPModelForTokenClassification)


MSE_MODEL_TYPE = 'mse'


@registry.register_model_config("MSEMLPModelConfig")
class MSEMLPModelConfig(BaseModelConfig):
    model_type = MSE_MODEL_TYPE

    def __init__(
            self,
            hidden_size: int = 256,
            hidden_act: str = 'relu',
            classifier_dropout: float = 0.0,
            hidden_droput: float = 0.0,
            num_labels: int = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.classifier_dropout = classifier_dropout
        self.hidden_dropout = hidden_droput
        self.num_labels = num_labels



class MLPForMSELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]

        self.fc1 = nn.Linear(config.hidden_size, 64)

        self.fc2 = nn.Linear(64, 8)

        self.fc3 = nn.Linear(8, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        output = torch.sigmoid(self.fc3(hidden_states))
        return output


@registry.register_model("MLPModelForMSELoss")
class MLPModelForMSELoss(BasePreTrainedModel):
    config_class = MSEMLPModelConfig
    base_model_prefix = CEL_MODEL_TYPE
    supports_gradient_checkpointing = False

    def __init__(self, config: MSEMLPModelConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.layer = MLPForMSELoss(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)

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

        hidden_states = torch.norm(inputs_embeds[:, :self.config.hidden_size], p=2, dim=-1, keepdim=True)

        logits = self.layer(hidden_states)

        loss = None
        if labels is not None:
            mse_loss_fn = nn.MSELoss()
            loss = mse_loss_fn(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )