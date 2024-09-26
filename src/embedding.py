import os
import logging
from typing import Any, Optional, List, Union

import numpy as np
import openai
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import tiktoken

from .base import BaseEmbedding
from .registry import registry

logger = logging.getLogger(__name__)

__all__ = ['OpenAIEmbedding', 'LinqEmbedding']

@registry.register_embedding('OpenAIEmbedding')
class OpenAIEmbedding(BaseEmbedding):

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.model is None:
            self.model = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')).embeddings

        if self.tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model(self.name_or_path)

        if self.max_length is None or self.max_length > 8192:
            logger.info(
                "Max length is not set or exceeds 8192, "
                "setting max length to 8192 as per OpenAI Embedding's maximum limit."
            )
            self.max_length = 8192

    def truncate(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be of type str, but got {type(text).__name__}.")

        max_token_length = self.max_length
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_token_length:
            truncated_tokens = tokens[:max_token_length]
            truncated_text = self.tokenizer.decode(truncated_tokens)
            return truncated_text
        return text

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
        """
        Computes text embeddings for given input texts.

        Args:
            texts (Union[str, List[str]]): The texts to embed.
            show_progress_bar (Optional[bool], optional): Whether to show a progress bar during encoding. Defaults to None.
            precision (Optional[str], optional): Precision for embeddings. Defaults to "float32".
            convert_to_numpy (bool, optional): If True, converts the embeddings to numpy arrays. Defaults to False.
            convert_to_tensor (bool, optional): If True, converts the embeddings to PyTorch tensor. Defaults to True.
            device (Optional[str], optional): The device to use for computation. Defaults to None.
            normalize_embeddings (bool, optional): If True, normalizes the embeddings. Defaults to False.

        Returns:
            Union[List[torch.Tensor], np.ndarray, torch.Tensor]: Computed embeddings.
        """

        if isinstance(texts, str):  # If a single text, convert it to list
            texts = [texts]

        truncated_texts = list(map(self.truncate, texts))

        if show_progress_bar is None:
            show_progress_bar = True  # Default to showing progress bar

        encoded_embeds = []
        for start_index in tqdm(
                range(0, len(truncated_texts), self.batch_size),
                desc=f"Encoding {self.batch_size} batches",
                disable=not show_progress_bar
        ):
            batch_input_texts = truncated_texts[start_index:start_index + self.batch_size]
            datas = self.model.create(input=batch_input_texts, model=self.name_or_path).data

            for data in datas:
                embed = torch.tensor(data.embedding, dtype=torch.float32).unsqueeze(0)
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


def last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class LinqEmbedding(BaseEmbedding):
    name_or_path: Optional[str] = 'Linq-AI-Research/Linq-Embed-Mistral'

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.name_or_path != 'Linq-AI-Research/Linq-Embed-Mistral':
            logger.warning(
                f"Invalid model name or path: {self.name_or_path}. "
                f"Set to 'Linq-AI-Research/Linq-Embed-Mistral'."
            )
            self.name_or_path = 'Linq-AI-Research/Linq-Embed-Mistral'
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.name_or_path,
                torch_dtype=torch.float16
            )
            self.model.to(self.device)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.name_or_path
            )

        if self.max_length is None or self.max_length > 32768:
            logger.info(
                "Max length is not set or exceeds 32768, "
                "setting max length to 32768 as positional embedding's maximum limit of Mistral Model."
            )
            self.max_length = 32768

    @torch.no_grad()
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
        """
        Computes text embeddings for given input texts.

        Args:
            texts (Union[str, List[str]]): The texts to embed.
            show_progress_bar (Optional[bool], optional): Whether to show a progress bar during encoding. Defaults to None.
            precision (Optional[str], optional): Precision for embeddings. Defaults to "float32".
            convert_to_numpy (bool, optional): If True, converts the embeddings to numpy arrays. Defaults to False.
            convert_to_tensor (bool, optional): If True, converts the embeddings to PyTorch tensor. Defaults to True.
            device (Optional[str], optional): The device to use for computation. Defaults to None.
            normalize_embeddings (bool, optional): If True, normalizes the embeddings. Defaults to False.

        Returns:
            Union[List[torch.Tensor], np.ndarray, torch.Tensor]: Computed embeddings.
        """

        if isinstance(texts, str):
            texts = [texts]

        if show_progress_bar is None:
            show_progress_bar = True

        # Sorting texts by length (optional for optimization)
        length_sorted_idx = np.argsort([len(t) for t in texts])[::-1]
        input_texts_sorted = [texts[idx] for idx in length_sorted_idx]

        encoded_embeds = []

        # Batch processing loop
        for start_index in tqdm(
                range(0, len(input_texts_sorted), self.batch_size),
                desc=f"Encoding {self.batch_size} batches",
                disable=not show_progress_bar
        ):
            batch_input_texts: List[str] = input_texts_sorted[start_index: start_index + self.batch_size]

            # Tokenize input texts
            batch_encoding = self.tokenizer(
                batch_input_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Move batch to GPU/CPU
            if device:
                batch_encoding = {key: value.to(device) for key, value in batch_encoding.items()}

            # Run the model to get outputs
            outputs = self.model(**batch_encoding)

            # Extract embeddings (e.g., with last token pooling)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_encoding['attention_mask'])

            # Normalize embeddings if required
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Adjust precision if required
            if precision and precision != "float32":
                embeddings = embeddings.to(dtype=getattr(torch, precision))

            # Detach from the graph and move to CPU if necessary
            embeddings = embeddings.detach()  # Batch processing using start_index and end_index
            encoded_embeds.append(embeddings)

        # Stack all embeddings from batches
        encoded_embeds = torch.cat(encoded_embeds, dim=0)

        # Convert to numpy or tensor based on options
        if convert_to_numpy:
            return encoded_embeds.cpu().numpy()
        elif convert_to_tensor:
            return encoded_embeds.to(device=device)
        else:
            return encoded_embeds.tolist()
