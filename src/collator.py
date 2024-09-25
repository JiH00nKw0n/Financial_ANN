from typing import List, Dict
from collections import defaultdict

import torch

from .base import BaseCollator


class CollatorForBinaryClassification(BaseCollator):

    def _process(self, inputs: List[Dict], **kwargs) -> Dict[str, torch.Tensor]:
        inputs = [i for i in inputs if all(value is not None for value in i.values())]

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


