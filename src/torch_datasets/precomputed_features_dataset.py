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

class PrecomputedFeaturesDataset(Dataset):
    def __init__(
        self, feature_extractor, feature_extractor_output_length, acc_dataset, device
    ):
        """

        Args:
            feature_extractor: Feature extractor to extract features from raw accelerometer data
            feature_extractor_output_length: Output size of the feature extractor
            acc_dataset: Appropriate AccDataset dataset instance for the feature extractor
            device: torch device to run feature extraction on

        """
        self.length = len(acc_dataset)

        dataloader = DataLoader(acc_dataset, batch_size=len(acc_dataset), shuffle=False, drop_last=False)
        i = 0
        feature_extractor.to(device)

        for data, label in (progress := tqdm(dataloader)):
            feature_extractor.eval()
            with torch.no_grad():
                data = data.to(device)
                features = feature_extractor(data)
                progress.set_description(
                    f"Extracting features from raw accelerometer data... {i}/{len(acc_dataset)}"
                )
                i = i + 1
                features_data = features.numpy(force=True)
                labels_data = label.numpy(force=True)
        self.features_data = features_data
        self.labels_data = labels_data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.from_numpy(self.features_data[idx]), torch.tensor(self.labels_data[idx], dtype=torch.long)