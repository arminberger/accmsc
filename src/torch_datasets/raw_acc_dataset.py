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


class RawAccDataset(Dataset):
    def __init__(
        self,
        motion_df_list,
        label_df_list,
        num_samples,
        prev_elements=0,
        motion_cols=["x", "y", "z"],
        label_col="label",
        transform=None,
        label_transform=None,
        channels_first=True,
        output_internal_list_index=False,
    ):
        """

        :param motion_df_list: List of motion dataframes, assumed to have an index containing timestamps and columns <motion_cols>
        :param label_df_list: List of labels, assumed to have an index containing timestamps and a label column with name <label_col>
        :param num_samples: Number of contiguous motion data samples per dataset element
        :param prev_elements: Number of elements to return per __getitem__ call
        :param timestamp_col:
        :param motion_cols:
        :param motion_chan:
        :param transform:
        :param label_transform:
        """
        # Check that motion_df_list and label_df_list have timestamps as index by checking that the first index element is a timestamp
        for i in range(len(motion_df_list)):
            assert is_datetime64_dtype(motion_df_list[i].index)
            assert is_datetime64_dtype(label_df_list[i].index)
        self.motion_df_list = motion_df_list
        self.label_df_list = label_df_list
        self.num_samples = num_samples
        self.prev_elements = prev_elements
        self.motion_cols = motion_cols
        self.motion_chan = len(motion_cols)
        self.label_col = label_col
        self.transform = transform
        self.label_transform = label_transform
        self.channels_first = channels_first
        # Calculate helper array for indexing
        self.first_index_list = []
        acc = 0
        for i in range(len(motion_df_list)):
            acc = (
                acc
                + (math.floor(motion_df_list[i].shape[0] / self.num_samples))
                - (self.prev_elements)
            )
            self.first_index_list.append(acc)
        self.length = acc

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        first_index = bisect.bisect(self.first_index_list, idx)
        second_index = (
            (idx - self.first_index_list[first_index - 1]) if first_index > 0 else idx
        ) + self.prev_elements
        # Get data prior to index as specified by prev_elements
        data = []
        for i in range(self.prev_elements):
            data.append(
                self.motion_df_list[first_index].iloc[
                    (second_index - (self.prev_elements - i))
                    * self.num_samples : (second_index + 1 - (self.prev_elements - i))
                    * self.num_samples,
                    :,
                ]
            )
        # Get data at index
        data.append(
            self.motion_df_list[first_index].iloc[
                second_index * self.num_samples : (second_index + 1) * self.num_samples,
                :,
            ]
        )
        # TODO: GET DATA AFTER INDEX

        # Get label corresponding to index
        # TODO: We assign label to first timestamp -- should not make meaningful difference
        data_timestamp = data[self.prev_elements].index[0]
        label_index = self.label_df_list[first_index].index.get_indexer(
            [data_timestamp], method="nearest"
        )[0]
        label = self.label_df_list[first_index][self.label_col].iloc[label_index]
        # Convert data to input ready for a model
        data = [elem[self.motion_cols].to_numpy() for elem in data]
        if self.channels_first:
            data = [elem.T for elem in data]

        if self.transform:
            # TODO: Not sure if this works
            data = list(map(self.transform, data))

        label = apply_label_transform(label, self.label_transform)

        # Final conversion to torch tensors
        # Turn data into tensor
        data = [torch.tensor(elem, dtype=torch.float32) for elem in data]
        data = torch.vstack(data)
        # Turn labels into tensor
        label = torch.tensor(label, dtype=torch.long)

        return data, label

def apply_label_transform(label, label_transform):
    if isinstance(label_transform, Mapping):
        label = label_transform[label]
    elif label_transform is not None:
        label = label_transform(label)
    return label