import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Extra
from typing import Any, Optional, Union, List, Dict
from collections import OrderedDict
import os
from tqdm import tqdm


class BaseEmbedding(BaseModel):
    name_or_path: Optional[str] = None
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    max_length: Optional[int] = None
    batch_size: Optional[int] = 128
    use_cache: Optional[bool] = True
    state_dict_cache: Optional[OrderedDict] = None
    cache_dir: Optional[str] = None
    save_path: Optional[str] = None
    device: Optional[str] = 'cuda' if torch.cuda.is_available() else 'cpu'

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        if self.use_cache:
            if self.state_dict_cache is None:
                self.state_dict_cache = OrderedDict()

            os.makedirs(os.path.join(self.cache_dir, self.name_or_path), exist_ok=True)

            self.save_path = os.path.join(self.cache_dir, self.name_or_path, f"{self.max_length}.pth")

            self.load_state_dict()

    def load_state_dict(self):
        if os.path.isfile(self.save_path):
            self.state_dict_cache = torch.load(self.save_path, map_location=torch.device(self.device))

    def save_state_dict(self):
        torch.save(self.state_dict_cache, self.save_path)

    def get_cached_embeddings(
            self,
            indices: Union[int, List[int]],
            show_progress_bar: Optional[bool] = None,
            precision: Optional[str] = "float32",
            convert_to_numpy: bool = False,
            convert_to_tensor: bool = True,
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:

        if isinstance(indices, int):  # If a single text, convert it to list
            indices = [indices]

        if show_progress_bar is None:
            show_progress_bar = True  # Default to showing progress bar

        encoded_embeds = []
        for idx in tqdm(
                indices,
                desc=f"Getting from state_dict_cache",
                disable=not show_progress_bar
        ):
            embed = self.state_dict_cache[idx].unsqueeze(0)
            encoded_embeds.append(embed)

        encoded_embeds = torch.cat(encoded_embeds, dim=0)  # Stack all embeddings along dimension 0

        # Optional: normalize embeddings if required
        if normalize_embeddings:
            encoded_embeds = torch.nn.functional.normalize(encoded_embeds, p=2, dim=1)

        # Convert precision if specified
        if precision and precision != "float32":
            encoded_embeds = encoded_embeds.to(dtype=getattr(torch, precision))

        # Convert to numpy or tensor based on options
        if convert_to_numpy:
            return encoded_embeds.cpu().numpy()
        elif convert_to_tensor:
            return encoded_embeds.to(device=device)
        else:
            return encoded_embeds.tolist()

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
            indices: Union[int, List[int]],
            show_progress_bar: Optional[bool] = None,
            precision: Optional[str] = "float32",
            convert_to_numpy: bool = False,
            convert_to_tensor: bool = True,
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
    ):
        if not self.use_cache or self.state_dict_cache is None:
            return self.get_embeddings(
                texts=texts,
                show_progress_bar=show_progress_bar,
                precision=precision,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                device=device,
                normalize_embeddings=normalize_embeddings
            )
        else:
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(indices, int):
                indices = [indices]

            if not len(indices) == len(texts):
                raise ValueError()

            indices_to_embed = []
            texts_to_embed = []

            for idx, text in zip(indices, texts):
                if idx not in self.state_dict_cache:
                    indices_to_embed.append(idx)
                    texts_to_embed.append(text)

            if len(texts_to_embed) != 0:
                embeddings_to_cache = self.get_embeddings(
                    texts=texts_to_embed,
                    show_progress_bar=show_progress_bar,
                    precision=precision,
                    convert_to_numpy=convert_to_numpy,
                    convert_to_tensor=convert_to_tensor,
                    device=device,
                    normalize_embeddings=normalize_embeddings
                )

                for i, idx in enumerate(indices_to_embed):
                    self.state_dict_cache[idx] = embeddings_to_cache[i]

                self.save_state_dict()


            return self.get_cached_embeddings(
                indices=indices,
                show_progress_bar=show_progress_bar,
                precision=precision,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                device=device,
                normalize_embeddings=normalize_embeddings
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
    model_config = ConfigDict(extra='forbid')

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
    model_config = ConfigDict(extra='forbid')

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
    model_config = ConfigDict(extra='forbid')

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