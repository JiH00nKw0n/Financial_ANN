from typing import List, Dict
from collections import defaultdict

import torch

from .base import BaseCollator


class CollatorForBinaryClassification(BaseCollator):

    def _process(self, inputs: List[Dict], **kwargs) -> Dict[str, torch.Tensor]:
        colums_to_check = ['text', 'ticker', 'event_start_at_et', 'd+1_open', 'd+2_open', 'd+3_open', 'd+1_close', 'd+2_close', 'd+3_close']

        # Filter inputs by checking only the values in colums_to_check
        inputs = [i for i in inputs if all(i[col] is not None for col in colums_to_check if col in i)]

        output = defaultdict()

        text_inputs: List[str] = [i['text'] for i in inputs]
        inputs_embeds = self.feature_extractor(text_inputs, **kwargs)
        output['inputs_embeds'] = inputs_embeds

        start = torch.FloatTensor([i['d+1_open'] for i in inputs])
        output['start'] = start
        end = torch.FloatTensor([i['d+3_close'] for i in inputs])
        output['end'] = end

        labels = start > end
        output['labels'] = labels

        return output


