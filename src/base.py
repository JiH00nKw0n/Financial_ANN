import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Extra
from typing import Any, Optional, Union, List, Dict


class BaseEmbedding(BaseModel):
    model_name_or_path: Optional[str] = None
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    max_length: Optional[int] = None
    batch_size: Optional[int] = 128

    def get_embeddings(
            self,
            texts: Union[str, List[str]],
            show_progress_bar: Optional[bool] = None,
            precision: Optional[str] = "float32",
            convert_to_numpy: bool = False,
            convert_to_tensor: bool = True,
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        raise NotImplementedError

    def __call__(
            self,
            texts: Union[str, List[str]],
            show_progress_bar: Optional[bool] = None,
            precision: Optional[str] = "float32",
            convert_to_numpy: bool = False,
            convert_to_tensor: bool = True,
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
    ):
        return self.get_embeddings(
            texts,
            show_progress_bar,
            precision,
            convert_to_numpy,
            convert_to_tensor,
            device,
            normalize_embeddings
        )


class BaseBuilder(BaseModel):
    split: Optional[str | List[str]] = None

    def build_datasets(self):
        raise NotImplementedError


class BaseCollator(BaseModel):
    feature_extractor: Optional[Any] = None
    show_progress_bar: Optional[bool] = True
    precision: Optional[str] = "float32"
    convert_to_numpy: bool = False
    convert_to_tensor: bool = True
    device: Optional[str] = 'cuda'
    normalize_embeddings: bool = True

    def __call__(self, inputs, **kwargs):
        kwargs = dict(
            {
                'show_progress_bar': self.show_progress_bar,
                'precision': self.precision,
                'convert_to_numpy': self.convert_to_numpy,
                'convert_to_tensor': self.convert_to_tensor,
                'device': self.device,
                'normalize_embeddings': self.normalize_embeddings,
            }, **kwargs
        )
        return self._process(inputs, **kwargs)

    def _process(self, inputs, **kwargs):
        raise NotImplementedError


class BaseTask(BaseModel):
    """
    The `BaseTask` is an abstract class that provides a template for task-specific classes. It defines methods for
    building models, processors, and datasets, which must be implemented by any class that inherits from `BaseTask`.

    This base class enforces consistency across tasks by requiring the implementation of core methods that set up
    the necessary components for task execution.
    """

    config: Any
    model_config = ConfigDict(extra=Extra.forbid)

    @classmethod
    def setup_task(cls, config: Any):
        """
        Initialize and return an instance of the task using the provided configuration.

        Args:
            config (Any): The configuration for the task.

        Returns:
            BaseTask: An instance of the task.
        """
        return cls(config=config)

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ):
        """
        Abstract method for building a model. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ):
        """
        Abstract method for building a processor. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
    ):
        """
        Abstract method for building datasets. Must be implemented by subclasses.
        """
        raise NotImplementedError


class BaseTrainTask(BaseTask):
    """
    The `BaseTask` is an abstract class that provides a template for task-specific classes. It defines methods for
    building models, processors, and datasets, which must be implemented by any class that inherits from `BaseTask`.

    This base class enforces consistency across tasks by requiring the implementation of core methods that set up
    the necessary components for task execution.
    """

    config: Any
    model_config = ConfigDict(extra=Extra.forbid)

    @classmethod
    def setup_task(cls, config: Any):
        """
        Initialize and return an instance of the task using the provided configuration.

        Args:
            config (Any): The configuration for the task.

        Returns:
            BaseTask: An instance of the task.
        """
        return cls(config=config)

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ):
        """
        Abstract method for building a model. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ):
        """
        Abstract method for building a processor. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
    ):
        """
        Abstract method for building datasets. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None,
    ):
        """
        Abstract method for building datasets. Must be implemented by subclasses.
        """
        raise NotImplementedError


class BaseEvaluateTask(BaseTask):
    """
    The `BaseTask` is an abstract class that provides a template for task-specific classes. It defines methods for
    building models, processors, and datasets, which must be implemented by any class that inherits from `BaseTask`.

    This base class enforces consistency across tasks by requiring the implementation of core methods that set up
    the necessary components for task execution.
    """

    config: Any
    model_config = ConfigDict(extra=Extra.forbid)

    @classmethod
    def setup_task(cls, config: Any):
        """
        Initialize and return an instance of the task using the provided configuration.

        Args:
            config (Any): The configuration for the task.

        Returns:
            BaseTask: An instance of the task.
        """
        return cls(config=config)

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ):
        """
        Abstract method for building a model. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ):
        """
        Abstract method for building a processor. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
    ):
        """
        Abstract method for building datasets. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_evaluator(
            self,
            evaluator_config: Optional[Dict] = None,
    ):
        """
        Abstract method for building datasets. Must be implemented by subclasses.
        """
        raise NotImplementedError

class BaseEvaluator(BaseModel):
    pass