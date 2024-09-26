import logging
import os
from typing import Optional, Dict, Type, TypeVar

from datasets import Dataset, IterableDataset, interleave_datasets, concatenate_datasets
from omegaconf import DictConfig
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)

from .registry import registry
from .base import BaseTrainTask
from .config import TrainConfig

ModelType = Type[PreTrainedModel]
DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = [
    "TrainTask",
]

logger = logging.getLogger(__name__)


class TrainTask(BaseTrainTask):
    config: TrainConfig

    def build_model(self, model_config: Optional[Dict] = None):
        """
        Builds a custom model using the provided configuration from the registry. If a LoRA configuration
        is provided for either text or vision models, it applies the configuration for parameter-efficient fine-tuning.

        Args:
            model_config (`Optional[Dict]`, *optional*):
                The model configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ModelType`: The custom model instance with optional LoRA fine-tuning.

        Raises:
            TypeError: If `model_config.lora` is not a valid `DictConfig` object.
        """
        model_config = model_config if model_config is not None else self.config.model_config.copy()

        # Get the model configuration and model class from the registry
        model_cfg_cls = registry.get_model_config_class(model_config.config_cls)
        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert model_cfg_cls is not None, "Model config {} not properly registered.".format(model_cfg_cls)

        # Initialize the model configuration and model
        model_cfg = model_cfg_cls(**model_config.config)
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

        return builder_cls(**dataset_config.config)

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
            eval_dataset=dataset['eval'],
            data_collator=collator,
        )
