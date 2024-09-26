from typing import Optional, List

from datasets import load_dataset, DatasetDict

from .base import BaseBuilder
from .registry import registry

@registry.register_builder('FinancialTranscriptBuilder')
class FinancialTranscriptBuilder(BaseBuilder):
    split: Optional[str | List[str]] = ['train', 'val']

    def build_datasets(self) -> DatasetDict:
        dataset_dict = DatasetDict()

        if isinstance(self.split, str):
            self.split = [self.split]

        for split in self.split:
            dataset_dict[split] = load_dataset('Linq-AI-Research/FinancialANN', split=split, token=True)

        return dataset_dict
