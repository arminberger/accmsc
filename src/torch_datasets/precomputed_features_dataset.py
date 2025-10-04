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
        self,
        feature_extractor,
        feature_extractor_output_length,
        acc_dataset,
        device,
        hr_noise_std=0.0,
        noise_seed=None,
    ):
        """

        Args:
            feature_extractor: Feature extractor to extract features from raw accelerometer data
            feature_extractor_output_length: Output size of the feature extractor
            acc_dataset: Appropriate AccDataset dataset instance for the feature extractor
            device: torch device to run feature extraction on

        """
        self.length = len(acc_dataset)
        self.hr_noise_std = float(hr_noise_std) if hr_noise_std is not None else 0.0
        self.noise_seed = noise_seed

        dataloader = DataLoader(acc_dataset, batch_size=len(acc_dataset), shuffle=False, drop_last=False)
        i = 0
        feature_extractor.to(device)

        hr_features_array = None
        for batch in (progress := tqdm(dataloader)):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                data, label, hr_tensor = batch
                hr_features_array = hr_tensor.numpy(force=True).astype(np.float32)
                if hr_features_array.ndim == 1:
                    hr_features_array = hr_features_array[:, None]
            else:
                data, label = batch
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
        if hr_features_array is not None:
            if self.hr_noise_std > 0:
                rng = np.random.default_rng(self.noise_seed)
                hr_noise = rng.normal(0.0, self.hr_noise_std, size=hr_features_array.shape).astype(np.float32)
                hr_features_array = hr_features_array + hr_noise
            features_data = np.concatenate([features_data, hr_features_array], axis=1)
        self.features_data = features_data
        self.labels_data = labels_data
        self.hr_features = hr_features_array
        self.base_feature_dim = feature_extractor_output_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.from_numpy(self.features_data[idx]), torch.tensor(self.labels_data[idx], dtype=torch.long)