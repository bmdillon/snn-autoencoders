import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class cdataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        """
        Args:
            data (numpy array or torch tensor): The input data of shape (N, 19)
            labels (numpy array or torch tensor, optional): The labels of shape (N,). Default is None.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        # Check if data is a numpy array or tensor and convert it to numpy if it's a tensor
        if isinstance(data, torch.Tensor):
            self.data = data.numpy()
        else:
            self.data = np.array(data)
        # convert back to a torch tensor after normalization
        self.data = torch.tensor(self.data, dtype=torch.float32)
        # optional labels
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get data and label if available
        sample = self.data[idx]
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        # apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        return sample, label