import torchvision
from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd


if __name__ == '__main__':
    train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                  train=True, download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST(root='./data/test/FashionMNIST',
                                                 train=False, download=True,
                                                 transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

    batch = next(iter(train_loader))

    images, labels = batch  # features and labels
    # images.shape = [10,1,28,28]
    # labels.shape = [10]
    print(images.size())