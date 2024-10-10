import logging
from typing import Optional, Dict, Type, TypeVar

import numpy as np
from datasets import Dataset, IterableDataset
from peft import LoraConfig
from transformers import (
    PreTrainedModel,
    TrainingArguments, EvalPrediction, AutoTokenizer,
)

from . import BaseEvaluateTask, load_yml
from .base import BaseTrainTask
from .config import TrainConfig, EvaluateConfig
from .registry import registry

ModelType = Type[PreTrainedModel]
DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = [
    "TrainTask",
]

logger = logging.getLogger(__name__)


def setup_task(config):
    assert "task" in config.run_config, "Task name must be provided."

    task_name = config.run_config.task
    task = registry.get_task_class(task_name).setup_task(config=config)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


# def compute_metrics(pred: EvalPrediction):
#     # Extract the logits (predictions) and labels from EvalPrediction
#     logits = pred.predictions
#     labels = pred.label_ids
#
#     # Apply the 0.5 threshold to logits to convert to binary predictions
#     predictions = (logits >= 0.5).astype(int)
#
#     # Calculate accuracy
#     accuracy = np.mean(predictions == labels)
#
#     return {"accuracy": accuracy}

def compute_metrics(pred: EvalPrediction):
    # Extract the logits (predictions) and labels from EvalPrediction
    logits = pred.predictions
    labels = pred.label_ids

    # Convert logits to predicted labels by taking the argmax (for multi-class classification)
    predictions = np.argmax(logits, axis=-1)

    # Calculate the overall accuracy (for all labels 0, 1, 2)
    overall_accuracy = np.mean(predictions == labels)

    # Initialize true positives, false positives, false negatives for each class
    true_positives = np.zeros(3)
    false_positives = np.zeros(3)
    false_negatives = np.zeros(3)

    for i in range(3):  # Assuming 3 classes: 0, 1, 2
        true_positives[i] = np.sum((predictions == i) & (labels == i))
        false_positives[i] = np.sum((predictions == i) & (labels != i))
        false_negatives[i] = np.sum((predictions != i) & (labels == i))

    # Calculate precision and recall for each class
    precision = true_positives / (true_positives + false_positives + 1e-8)  # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    # Calculate F1 score for each class
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Macro F1 score (average F1 across all classes)
    f1_score_macro = np.mean(f1_per_class)

    # Calculate accuracy for only 0 and 2 labels (ignoring label 1)
    mask_02 = (labels == 0) | (labels == 2)
    accuracy_02 = np.mean(predictions[mask_02] == labels[mask_02])

    return {
        "overall_accuracy": overall_accuracy,
        "f1_score_macro": f1_score_macro,
        "accuracy_pos&neg": accuracy_02
    }

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

        if model_config.lora is not None:
            lora_config_path = model_config.lora

            # Apply LoRA configuration to text model
            if lora_config_path is not None:
                lora_config = load_yml(lora_config_path)
                peft_config = LoraConfig(**lora_config)
                model.set_model_to_lora(peft_config)

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
            compute_metrics=compute_metrics,
        )


@registry.register_task('LinqTrainTask')
class LinqTrainTask(BaseTrainTask):

    def build_collator(
            self,
            collator_config: Optional[Dict] = None
    ):
        collator_config = collator_config if collator_config is not None else self.config.collator_config

        collator_name = collator_config.cls_name
        collator_cls = registry.get_collator_class(collator_name)

        assert collator_cls is not None, "Collator {} not properly registered.".format(collator_name)

        return collator_cls(
            feature_extractor=AutoTokenizer.from_pretrained(collator_config.pretrained_model_name_or_path),
            **collator_config.config
        )


@registry.register_task('EvaluateTask')
class EvaluateTask(BaseEvaluateTask):
    config: EvaluateConfig

    def build_model(self, model_config: Optional[Dict] = None):
        model_config = model_config if model_config is not None else self.config.model_config.copy()

        model_cls = registry.get_model_class(model_config.model_cls_name)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)

        model = model_cls(**model_config.config)

        return model.cuda().eval()

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

    def build_evaluator(
            self,
            evaluator_config: Optional[Dict] = None
    ):
        evaluator_config = evaluator_config if evaluator_config is not None else self.config.evaluator_config

        assert "runner" in self.config.run_config, "Evaluator name must be provided."

        evaluator_name = self.config.run_config.runner
        evaluator_cls = registry.get_evaluator_class(evaluator_name)
        assert evaluator_cls is not None, "Trainer {} not properly registered.".format(evaluator_name)

        model = self.build_model()
        collator = self.build_collator()
        dataset = self.build_datasets()

        return evaluator_cls(
            model=model,
            dataset=dataset['test'],
            data_collator=collator,
            **evaluator_config.config
        )
