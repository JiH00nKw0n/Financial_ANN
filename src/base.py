import numpy as np
import torch
from pydantic import BaseModel
from typing import Any, Optional, Union, List


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
            normalize_embeddings: bool = False,
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
            normalize_embeddings: bool = False,
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
