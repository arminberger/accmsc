from src.train_ssl import gen_aug
import numpy as np
import torch
from torch.utils.data import Dataset

class SSLazyAugDatasetSubjectWise(Dataset):
    def __init__(
        self,
        dataset,
        aug,
        device=None,
        channels_first=True,
        memoization=True,
    ):
        """

        Args:
            dataset: SSRawDataset with channels_first = False
            aug:
            motion_cols:
            channels_first:
            memoization:
        """
        self.raw_dataset = dataset
        self.aug = aug
        self.my_channels_first = channels_first
        self.memoization = memoization
        if self.memoization and len(self.raw_dataset) > 0:
            if self.my_channels_first:
                self.cache = torch.full(
                    (
                        len(self.raw_dataset),
                        self.raw_dataset[0].shape[1],
                        self.raw_dataset[0].shape[0],
                    ),
                    -1024,
                    dtype=torch.float32,
                )
            else:
                self.cache = torch.full(
                    (
                        len(self.raw_dataset),
                        self.raw_dataset[0].shape[0],
                        self.raw_dataset[0].shape[1],
                    ),
                    -1024,
                    dtype=torch.float32,
                )
            #self.cache = self.cache.to(self.device)
        self.cache_mask = np.zeros(len(self.raw_dataset), dtype=bool)
    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        if self.memoization and self.cache_mask[idx]:
            item = self.cache[idx]
        else:
            item = self.raw_dataset.__getitem__(idx)
            item = item.unsqueeze(0)
            item = gen_aug(item, self.aug)
            item = item.squeeze(0)
            item = item.type(torch.float32)
            if self.memoization:
                if self.my_channels_first:
                    self.cache[idx] = torch.transpose(item.type(torch.float32), 0, 1)
                    item = self.cache[idx]
                else:
                    self.cache[idx] = item.type(torch.float32)

                self.cache_mask[idx] = True


        return item
