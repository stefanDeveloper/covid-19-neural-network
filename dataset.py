import torch


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, images, labels):
        """Initialization"""
        self.labels = labels
        self.images = images

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        X = self.images[index]
        y = self.labels[index]

        return X, y
