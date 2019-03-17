import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                              train=True,download=True,
                                              transform=transforms.Compose(transforms.ToTensor()))
