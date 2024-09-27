# Financial_ANN

# Train

``` shell
bash ./scripts/train.sh
```

train.py
``` python
def main() -> None:
    ...
    # train.sh에서 지정한 config 파일을 load
    train_cfg = TrainConfig(**load_yml(args.cfg_path)) 
    init_distributed_mode(args)
    setup_seeds(train_cfg.run_config.seed)
    setup_logger()

    wandb.login(key=args.wandb_key)

    # config에 따라 task를 로드 (`setup_task` function from `src/task.py`)
    task = setup_task(train_cfg)

    # Huggingface Trainer를 상속 받는 Trainer class를 load
    trainer = task.build_trainer()
    trainer.add_callback(CustomWandbCallback())
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    wandb.finish()
    trainer.save_model()
```

Task.py
```python
trainer = task.build_trainer()
```
여기서 build_trainer의 역할은 task라는 class에 저장된 config 변수에 따라 `model`, `embedding(feature 추출)`, `collator`, `dataset`을 불러와 Trainer를 선언하는 역할.
각각 `build_model`, `build_embedding`, `build_collator`, `build_dataset`이라는 함수를 사용하여 load함

``` python
@registry.register_task('TrainTask')
class TrainTask(BaseTrainTask):
    config: TrainConfig

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config if model_config is not None else self.config.model_config.copy()

        # Get the model configuration and model class from the registry
        config_cls = registry.get_model_config_class(model_config.config_cls_name)
        model_cls = registry.get_model_class(model_config.model_cls_name)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert config_cls is not None, "Model config {} not properly registered.".format(config_cls)

        # Initialize the model configuration and model
        model_cfg = config_cls(**model_config.config)
        model = model_cls(model_cfg)

        return model.cuda().train()


    def build_embedding(
            self,
            embedding_config: Optional[Dict] = None
    ):
        embedding_config = embedding_config if embedding_config is not None else self.config.embedding_config

        embedding_name = embedding_config.cls_name
        embedding_cls = registry.get_embedding_class(embedding_name)
        assert embedding_cls is not None, "Embedding {} not properly registered.".format(embedding_name)

        return embedding_cls(**embedding_config.config)

    def build_collator(
            self,
            collator_config: Optional[Dict] = None
    ):
        collator_config = collator_config if collator_config is not None else self.config.collator_config

        collator_name = collator_config.cls_name
        collator_cls = registry.get_collator_class(collator_name)

        assert collator_cls is not None, "Collator {} not properly registered.".format(collator_name)

        feature_extractor = self.build_embedding()

        return collator_cls(
            feature_extractor=feature_extractor,
            **collator_config.config
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
    ):
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        builder_name = dataset_config.cls_name
        builder_cls = registry.get_builder_class(builder_name)

        assert builder_cls is not None, "Builder {} not properly registered.".format(builder_name)

        return builder_cls(**dataset_config.config).build_datasets()

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        trainer_config = trainer_config if trainer_config is not None else self.config.trainer_config

        assert "runner" in self.config.run_config, "Trainer name must be provided."

        trainer_name = self.config.run_config.runner
        trainer_cls = registry.get_trainer_class(trainer_name)
        assert trainer_cls is not None, "Trainer {} not properly registered.".format(trainer_name)

        model = self.build_model()
        collator = self.build_collator()
        dataset = self.build_datasets()

        return trainer_cls(
            model=model,
            args=TrainingArguments(**trainer_config),
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            data_collator=collator,
        )
```
model
``` python
## 학습 모델 내부의 2-hidden layer를 가지는 MLP layer
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

## 학습 모델은 해당 layer를 1개 이상 가지는 모델로 설정하여 depth를 조절하여 사용하고 있음
class MLPModelForTokenClassification(BasePreTrainedModel):
    config_class = MLPModelConfig
    base_model_prefix = MODEL_TYPE
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

```
