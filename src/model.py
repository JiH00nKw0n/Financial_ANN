import os
from typing import Optional, Union, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import AutoModel, AutoConfig, PretrainedConfig, PreTrainedModel, Cache
from transformers.activations import ACT2FN
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutputWithPast
from peft import get_peft_model, LoraConfig

from .base import BaseModelConfig, BasePreTrainedModel
from .registry import registry

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


def pool(last_hidden_state: torch.Tensor,
         attention_mask: torch.Tensor,
         pool_type: str) -> torch.Tensor:
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weighted_avg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
        s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        emb = s / d
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


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

        sliced_embed = inputs_embeds[:, :self.config.hidden_size].detach()

        hidden_states = sliced_embed / sliced_embed.norm(p=2, dim=-1, keepdim=True)
        hidden_states = hidden_states.detach()

        logits = self.layer(hidden_states)

        loss = None
        if labels is not None:
            mse_loss_fn = nn.MSELoss()
            loss = mse_loss_fn(logits, labels.float().unsqueeze(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )


@registry.register_model_config("LinqConfig")
class LinqConfig(PretrainedConfig):
    model_type = "linq"

    def __init__(
            self,
            model_name_or_path: Union[str, os.PathLike] = 'Linq-AI-Research/Linq-Embed-Mistral',
            pool_type: str = 'last',
            num_labels: int = 3,
            problem_type: str = 'single_label_classification',
            initializer_range: float = 0.02,
            hidden_size: int = 4096,
            hidden_act: str = 'relu',
            **kwargs
    ):
        if kwargs.get('torch_dtype') == 'fp16':
            kwargs['torch_dtype'] = torch.float16

        model_config = kwargs.pop('model_config', None)

        super().__init__(**kwargs)

        self.model_config = model_config if model_config is not None else dict(
            {"pretrained_model_name_or_path": model_name_or_path}, **kwargs
        )

        self.pool_type = pool_type
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act


class LinqMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]

        self.fc1 = nn.Linear(config.hidden_size, 1024)

        self.fc2 = nn.Linear(1024, 256)

        self.fc3 = nn.Linear(256, 64)

        self.fc4 = nn.Linear(64, 16)

        self.fc5 = nn.Linear(16, config.num_labels)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc3(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc4(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        output = self.fc5(hidden_states)
        return output


class LinqPreTrainedModel(PreTrainedModel):
    config_class = LinqConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@registry.register_model("LinqForSequenceClassification")
class LinqForSequenceClassification(LinqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.score = LinqMLP(config)

        # Initialize weights and apply final processing
        super().init_weights()

        self.model = AutoModel.from_pretrained(**config.model_config)

        super()._backward_compatibility_gradient_checkpointing()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_model_to_lora(self, config: LoraConfig):
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        pooled_output = pool(
            last_hidden_state=hidden_states,
            attention_mask=attention_mask,
            pool_type=self.config.pool_type
        )

        pooled_logits = self.score(pooled_output)

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )