import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def filter_dataset(data):
    indices = []
    for i in range(data.__len__()):
        if data.targets[i] in [0,1]:
            indices.append(i)
    return Subset(data, indices), indices


class Dataset:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to PyTorch tensors
            transforms.Normalize((0.5,), (0.5,))  # Normalizes the dataset
        ])


        self.train_data = datasets.MNIST(
            root='data', 
            train=True, 
            download=True, 
            transform=self.transform
        )


        self.test_data = datasets.MNIST(
            root='data', 
            train=False, 
            download=True, 
            transform=self.transform
        )

        self.train_subset, self.train_indices = filter_dataset(self.train_data)
        self.test_subset, self.test_indices = filter_dataset(self.test_data)
        self.train_targets = self.train_data.targets[self.train_indices]
        self.test_targets = self.test_data.targets[self.test_indices]

