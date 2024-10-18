from typing import List, Dict
from collections import defaultdict

import torch
from transformers import BatchEncoding

from .base import BaseCollator
from .registry import registry

@registry.register_collator('CollatorForBinaryClassification')
class CollatorForBinaryClassification(BaseCollator):

    def _process(self, inputs: List[Dict], **kwargs) -> Dict[str, torch.Tensor]:
        colums_to_check = ['text', 'ticker', 'event_start_at_et', 'd+1_open', 'd+2_open', 'd+3_open', 'd+1_close', 'd+2_close', 'd+3_close']

        # Filter inputs by checking only the values in colums_to_check
        inputs = [i for i in inputs if all(i[col] is not None for col in colums_to_check if col in i)]

        output = defaultdict()
        text_inputs: List[str] = [i['text'] for i in inputs]
        indices_inputs: List[int] = [int(i['#']) for i in inputs]
        inputs_embeds = self.feature_extractor(
            texts=text_inputs, indices=indices_inputs, **kwargs
        ).detach().to(self.device)
        output['inputs_embeds'] = inputs_embeds

        start = torch.FloatTensor([i['d+1_open'] for i in inputs])
        # output['start'] = start
        end = torch.FloatTensor([i['d+3_close'] for i in inputs])
        # output['end'] = end

        labels = start < end
        output['labels'] = labels.long().to(self.device)

        return output

@registry.register_collator('CollatorForClassificationWithNeutral')
class CollatorForClassificationWithNeutral(BaseCollator):
    max_length: int = 4096
    padding: bool = True
    truncation: bool = True
    return_tensors: str = 'pt'

    def __call__(self, inputs, **kwargs):
        kwargs = dict(
            {
                "max_length": self.max_length,
                "padding": self.padding,
                "truncation": self.truncation,
                "return_tensors": self.return_tensors,
            }, **kwargs
        )
        return self._process(inputs, **kwargs)

    def _process(self, inputs: List[Dict], **kwargs) -> BatchEncoding:
        colums_to_check = ['text', 'ticker', 'event_start_at_et', 'd+1_open', 'd+2_open', 'd+3_open', 'd+1_close',
                           'd+2_close', 'd+3_close']

        # Filter inputs by checking only the values in columns_to_check
        inputs = [i for i in inputs if all(i[col] is not None for col in colums_to_check if col in i)]

        texts = [i['text'] for i in inputs]
        batch_dict: BatchEncoding = self.feature_extractor(texts, **kwargs)

        start = torch.FloatTensor([i['d+1_open'] for i in inputs])
        end = torch.FloatTensor([i['d+3_close'] for i in inputs])

        # Calculate the percentage change between d+1_open and d+3_close
        percentage_change = (end - start) / start * 100

        # Assign labels based on the percentage change
        labels = torch.where(percentage_change >= 3, 2, torch.where(percentage_change <= -3, 0, 1))

        batch_dict.data['labels'] = labels.long().to(self.device)

        return batch_dict