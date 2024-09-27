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
