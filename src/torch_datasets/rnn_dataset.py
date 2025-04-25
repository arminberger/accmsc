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

# Subclass of torch.utils.data.Dataset for loading data from csv files
class SleepStagesDatasetRNN(Dataset):
    def __init__(
        self, data_np_list, rnn_sequence_length, transform=None, label_transform=None
    ):
        """

        :param data_np_list: List containing one numpy array per subject of shape (num_samples, num_features+2), where the last two columns are the label and the subject id
        :param rnn_sequence_length: Length of the sequence of samples that is fed into the RNN
        :param transform:
        :param label_transform:
        """
        # data_numpy has dimensions (num_subjects, num_samples, num_features+2)
        self.data_np_list = data_np_list
        self.transform = transform
        self.label_transform = label_transform
        self.rnn_sequence_length = rnn_sequence_length

        self.first_index_list = []
        acc = 0
        for i in range(len(data_np_list)):
            # Assumes rnn_sequence_length is smaller than the number of samples per subject
            acc = acc + data_np_list[i].shape[0] - (self.rnn_sequence_length - 1)
            self.first_index_list.append(acc)

    def __len__(self):
        length = 0
        for i in range(len(self.data_np_list)):
            length = (
                length + self.data_np_list[i].shape[0] - (self.rnn_sequence_length - 1)
            )
        return length

    def __getitem__(self, idx):
        first_index = bisect.bisect(self.first_index_list, idx)
        second_index = (
            (idx - self.first_index_list[first_index - 1]) if first_index > 0 else idx
        )
        data = []

        for i in range(self.rnn_sequence_length):
            data.append(self.data_np_list[first_index][second_index + i, 0:1024])

        label = self.data_np_list[first_index][
            second_index + self.rnn_sequence_length - 1, 1024
        ]
        # label = label + 1  # 1 is always added to labels to avoid -1 labels

        if self.transform:
            data = list(map(self.transform, data))

        label = apply_label_transform(label, self.label_transform)

        # Turn data into tensor
        for i in range(len(data)):
            data[i] = torch.tensor(data[i], dtype=torch.float32)
        data = torch.vstack(data)
        # Turn labels into tensor
        label = torch.tensor(label, dtype=torch.long)

        return data, label

