import bisect
import numbers

import torch
from torch.utils.data import Dataset, DataLoader
import math
from collections.abc import Mapping
import numpy as np
from tqdm import tqdm
import pandas as pd
from pandas.api.types import is_datetime64_dtype

class ListDatasetPadded(Dataset):
    def __init__(self, dataset_list, sequence_length=1):
        '''

        Args:
            dataset_list: Assumed to be a list of instances (e.g. subjects) of the same dataset class
            prev_elements:
            post_elements: Should always be 0
            pad_sequence: If True,
        '''
        self.dataset_list = dataset_list
        self.sequence_length = sequence_length
        # Calculate helper array for indexing
        self.first_index_list = []
        acc = 0
        for i in range(len(dataset_list)):
            acc = acc + len(dataset_list[i])
            self.first_index_list.append(acc)
        self.length = acc
        self.data_is_tensor = torch.is_tensor(dataset_list[0][0][0])
        self.label_is_tensor = torch.is_tensor(dataset_list[0][0][1])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        first_index = bisect.bisect(self.first_index_list, idx)
        second_index = (
            (idx - self.first_index_list[first_index - 1]) if first_index > 0 else idx
        )
        sequence_first_index = max(0, second_index - self.sequence_length + 1)
        sequence_last_index = second_index
        data = []
        label = self.dataset_list[first_index][sequence_last_index][1]
        for i in range(sequence_first_index, sequence_last_index + 1):
            data.append(self.dataset_list[first_index][i][0])
        # Now pad with zeros if necessary
        seq_length = len(data)
        if len(data) < self.sequence_length:
            for i in range(self.sequence_length - len(data)):
                if self.data_is_tensor:
                    curr_data = torch.zeros_like(self.dataset_list[first_index][0][0])
                else:
                    curr_data = np.zeros_like(self.dataset_list[first_index][0][0])
                data.append(curr_data)
        if not self.data_is_tensor:
            data = [torch.tensor(elem, dtype=torch.float32) for elem in data]
        data = torch.vstack(data)
        if not self.label_is_tensor:
            label = torch.tensor(label, dtype=torch.long)
        return data, label, seq_length