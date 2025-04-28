import bisect
import torch
from torch.utils.data import Dataset
import math

class SSRawDataset(Dataset):
    def __init__(
        self,
        motion_df_list,
        win_length,
        motion_cols=["x", "y", "z"],
        transform=None,
        channels_first=True,
    ):
        """

        :param motion_df_list: List of motion dataframes, assumed to have a timestamp column with name <timestamp_col>
        :param num_samples: Number of contiguous motion data samples per dataset element
        :param motion_cols:
        :param motion_chan:
        :param transform:
        :param label_transform:
        """
        super().__init__()
        self.motion_df_list = motion_df_list
        self.win_length = win_length
        self.motion_cols = motion_cols
        self.motion_chan = len(motion_cols)
        self.transform = transform
        self.channels_first = channels_first
        # Calculate helper array for indexing
        self.first_index_list = []
        acc = 0
        for i in range(len(motion_df_list)):
            acc = acc + (math.floor(motion_df_list[i].shape[0] / self.win_length))
            self.first_index_list.append(acc)
        # Calculate length of dataset
        self.length = 0
        for i in range(len(self.motion_df_list)):
            self.length = self.length + math.floor(
                self.motion_df_list[i].shape[0] / self.win_length
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Target and Domain not needed for self supervised learning
        target = 0
        domain = 0

        first_index = bisect.bisect(self.first_index_list, idx)
        second_index = (
            (idx - self.first_index_list[first_index - 1]) if first_index > 0 else idx
        )
        # Get data corresponding to index
        data = self.motion_df_list[first_index].iloc[
            second_index * self.win_length : (second_index + 1) * self.win_length, :
        ]
        # Convert data to input ready for a model
        data = data[self.motion_cols].to_numpy()
        if self.channels_first:
            data = data.T

        if self.transform:
            # TODO: Not sure if this works
            data = list(map(self.transform, data))

        # Final conversion to torch tensors
        # Turn data into tensor
        data = torch.tensor(data, dtype=torch.float32)

        return data
