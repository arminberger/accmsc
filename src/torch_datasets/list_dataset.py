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

class ListDataset(Dataset):
    def __init__(self, dataset_list, pad_sequence, prev_elements=0, post_elements=0):
        '''

        Args:
            dataset_list: Assumed to be a list of instances (e.g. subjects) of the same dataset class
            prev_elements:
            post_elements:
            pad_sequence: If True,
        '''
        self.dataset_list = dataset_list
        self.prev_elements = prev_elements
        self.post_elements = post_elements
        self.pad_sequence = pad_sequence
        # Calculate helper array for indexing
        self.first_index_list = []
        acc = 0
        for i in range(len(dataset_list)):
            if pad_sequence:
                acc = acc + len(dataset_list[i])
            elif len(dataset_list[i]) - self.prev_elements - self.post_elements < 0:
                acc = acc
            else:
                acc = acc + len(dataset_list[i]) - self.prev_elements - self.post_elements
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
        ) + self.prev_elements
        if self.pad_sequence:
            second_index = second_index - self.prev_elements
        data = []
        label = -1
        for i in range(self.prev_elements + self.post_elements + 1):
            # Check if the second_index is out of bounds for first_index
            if (second_index - self.prev_elements + i >= len(self.dataset_list[first_index]) or second_index - self.prev_elements + i < 0) and self.pad_sequence:
                # If padding is enabled, we return a zero tensor of the appropriate shape
                if self.data_is_tensor:
                    curr_data = torch.zeros_like(self.dataset_list[first_index][0][0])
                else:
                    curr_data = np.zeros_like(self.dataset_list[first_index][0][0])
                curr_label = -1
            else:
                curr_data, curr_label = self.dataset_list[first_index][second_index - self.prev_elements + i]
            data.append(curr_data)
            if i == self.prev_elements:
                label = curr_label
        if not self.data_is_tensor:
            data = [torch.tensor(elem, dtype=torch.float32) for elem in data]

        data = torch.vstack(data)
        if not self.label_is_tensor:
            label = torch.tensor(label, dtype=torch.long)
        return data, label