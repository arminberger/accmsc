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

class AccDataset(Dataset):
    def __init__(
        self,
        motion_df,
        label_df,
        samples_per_window,
        motion_cols=["x", "y", "z"],
        label_col="label",
        label_transform=None,
        channels_first=True,
        heart_rate_df=None,
        heart_rate_col="bpm",
    ):
        """
        Dataset for accelerometer data contiguously sampled at a fixed frequency.
        Args:
            motion_df: motion dataframe, assumed to have an index containing timestamps and columns <motion_cols>
            label_df: label dataframe, assumed to have an index containing timestamps and a label column with name <label_col>
            samples_per_window: How many contiguous motion data samples per dataset element
            motion_cols: motion_df columns to use as motion data
            label_col: label_df column to use as label
            transform:
            label_transform:
            channels_first:
        """
        assert is_datetime64_dtype(motion_df.index)
        assert is_datetime64_dtype(label_df.index)
        self.motion_df = motion_df
        self.label_df = label_df
        self.num_samples = samples_per_window
        self.motion_cols = motion_cols
        self.motion_chan = len(motion_cols)
        self.label_col = label_col
        self.label_transform = label_transform
        self.channels_first = channels_first
        self.heart_rate_series = None
        if heart_rate_df is not None and not heart_rate_df.empty:
            if not is_datetime64_dtype(heart_rate_df.index):
                raise ValueError("Heart rate dataframe index must be datetime64")
            if heart_rate_col not in heart_rate_df.columns:
                raise ValueError(f"Heart rate dataframe must contain column '{heart_rate_col}'")
            heart_rate_series = heart_rate_df[heart_rate_col].astype(np.float32)
            heart_rate_series = heart_rate_series[~heart_rate_series.index.duplicated(keep="first")]
            self.heart_rate_series = heart_rate_series.sort_index()

        self.length = math.floor(motion_df.shape[0] / samples_per_window)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.motion_df.iloc[
            idx * self.num_samples : (idx + 1) * self.num_samples,
            :,
        ]
        data_timestamp = sample.index[0]
        label_index = self.label_df.index.get_indexer(
            [data_timestamp], method="nearest"
        )[0]
        label = self.label_df[self.label_col].iloc[label_index]
        sample = sample[self.motion_cols].to_numpy(dtype=np.float32)
        if self.channels_first:
            sample = sample.T
        label = apply_label_transform(label, self.label_transform)

        hr_tensor = None
        if self.heart_rate_series is not None and len(self.heart_rate_series) > 0:
            hr_index = self.heart_rate_series.index.get_indexer([data_timestamp], method="nearest")
            if hr_index.size > 0 and hr_index[0] != -1:
                hr_value = float(self.heart_rate_series.iloc[hr_index[0]])
                hr_tensor = torch.tensor(hr_value, dtype=torch.float32)

        if hr_tensor is not None:
            return (
                torch.from_numpy(sample),
                torch.tensor(label, dtype=torch.long),
                hr_tensor,
            )

        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.long)

def apply_label_transform(label, label_transform):
    if isinstance(label_transform, Mapping):
        label = label_transform[int(label)]
    elif label_transform is not None:
        label = label_transform(label)
    return label






