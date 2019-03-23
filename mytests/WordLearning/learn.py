import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from wordprep import *
import os


device = torch.device('cpu')

train_x,train_y,test_x,test_y = create_sets('/pos-neg-sentdex/pos.txt','/pos-neg-sentdex/neg.txt')

NODE1 = 1500
NODE2 = 3000
NODE3 = 6000
CLASSES = 2
BATCH_SIZE = 100

W = len(train_x[0])
K = 1
P = 0
S = 1

outsizeofnn = (W-K+2*P)/S+1
print(W)
print(outsizeofnn)

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=len(train_x[0]), out_channels=NODE1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=NODE1, out_channels=NODE2, kernel_size=1)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.reshape(x.size(0),-1)
        x = x.view(-1, self.num_flat_features(x))  # flattening to switch to linear
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features