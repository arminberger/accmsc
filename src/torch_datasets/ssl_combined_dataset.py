from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, datasets) -> None:
        """
        This dataset is used to combine multiple datasets, each returning an augmented version of some basic dataset,
        into a single dataset.
        Args:
            datasets:
        """
        super().__init__()
        self.datasets = datasets
        try:
            for dataset in datasets:
                assert len(dataset) == len(datasets[0])
        except AssertionError:
            print("Datasets must be of equal length and have a one-to-one mapping")

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        return tuple([dataset[index] for dataset in self.datasets])