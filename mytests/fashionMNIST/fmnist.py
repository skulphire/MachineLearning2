import torchvision
from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd

SHUFFLE = False
BATCH_SIZE = 10
LR = 0.005 #learning rate
Train_epoch = 5

CLASS_CLOTHING = {0 :'T-shirt/top',
                  1 :'Trouser',
                  2 :'Pullover',
                  3 :'Dress',
                  4 :'Coat',
                  5 :'Sandal',
                  6 :'Shirt',
                  7 :'Sneaker',
                  8 :'Bag',
                  9 :'Ankle boot'}

device = torch.device('cpu')

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),1)
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.reshape(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train():
    model = Network().to(device)
    opti = torch.optim.Adam(model.parameters(),lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1,Train_epoch+1):
        for (feature,label) in next(iter(train_loader)): #image and label
            label, feature = label.to(device),feature.to(device)
            output = model(feature)
            loss = criterion(output,label)
            opti.zero_grad()
            loss.backward()
            opti.step()
            print("##########################")
            print("batch: "+batchID)
            print("##########################")
            if batchID % 1000 == 0:
                print('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, Train_epoch))
    return model

if __name__ == '__main__':
    train_set = torchvision.datasets.FashionMNIST(root='./data',
                                                  train=True, download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST(root='./data/test',
                                                 train=False, download=True,
                                                 transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    print(train_loader)

    batch = next(iter(train_loader))
    print(batch[1])
    batch = next(iter(train_loader))
    print(batch[1])
    batch = next(iter(train_loader))
    print(batch[1])
    batch = next(iter(train_loader))
    print(batch[1])
    batch = next(iter(train_loader))
    print(batch[1])

    #images, labels = batch  # features and labels
    # images.shape = [10,1,28,28]
    # labels.shape = [10]

    #w = images.reshape(images.size(0),-1)
    #print(w)

    #print(batch[2]) # out of range
    #print(images[0])
    #print(images[2])

    ############ start training and testing
    #model = train()



