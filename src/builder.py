from typing import Optional, List
from datetime import datetime

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

@registry.register_builder('MergedFinancialTranscriptBuilder')
class MergedFinancialTranscriptBuilder(BaseBuilder):
    split: str = 'train'
    train_years: Optional[List[int]] = None
    val_years: Optional[List[int]] = None

    def build_datasets(self) -> DatasetDict:
        dataset_dict = DatasetDict()

        dataset = load_dataset('Linq-AI-Research/FinancialANN_merged', split=self.split, token=True)

        def convert_to_date(example):
            if example['event_start_at_et']:
                example['event_start_at_et'] = datetime.strptime(example['event_start_at_et'],
                                                                 "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d")
            return example

        # 변환 적용
        dataset = dataset.map(convert_to_date)

        def filter_by_years(example, years):
            # 연도를 비교하여 리스트에 포함되어 있으면 True 반환
            return datetime.strptime(example['event_start_at_et'], "%Y-%m-%d").year in years


        train_dataset = dataset.filter(lambda example: filter_by_years(example, set(self.train_years)))
        val_dataset = dataset.filter(lambda example: filter_by_years(example, set(self.val_years)))

        dataset_dict['train'] = train_dataset
        dataset_dict['val'] = val_dataset

        return dataset_dict