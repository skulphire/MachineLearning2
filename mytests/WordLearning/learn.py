import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from wordprep import createdataset
import os


device = torch.device('cpu')

train_x,train_y,test_x,test_y = createdataset()

NODE1 = 1000
NODE2 = 2000
NODE3 = 3000
CLASSES = 2
BATCH_SIZE = 100
LR = 0.005 #learning rate
NUM_EPOCHS = 1

W = len(train_x[0])
K = 1
P = 0
S = 1

outsizeofnn = ((W+2*P-K)/S)+1
print(outsizeofnn)

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=len(train_x[0]), out_channels=NODE1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=NODE1, out_channels=NODE2, kernel_size=1)

        self.fc1 = nn.Linear(in_features=NODE2, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.out = nn.Linear(in_features=50, out_features=CLASSES)

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

def train(epochs = NUM_EPOCHS):
    model = Network().to(device)
    opti = torch.optim.Adam(model.parameters(),lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1,epochs+1):
        for batchID, (feature,label) in enumerate(train_x):
            label,feature = label.to(device),feature.to(device)
            output = model(feature)
            oss = criterion(output,label)
            opti.zero_grad()
            loss.backward()
            opti.step()
            if batchID % 1000 == 0:
                print('Loss X :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, epochs))
        for batchID, (feature,label) in enumerate(train_Y):
            label,feature = label.to(device),feature.to(device)
            output = model(feature)
            oss = criterion(output,label)
            opti.zero_grad()
            loss.backward()
            opti.step()
            if batchID % 1000 == 0:
                print('Loss Y :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, epochs))
    return model

if __name__ == '__main__':
    model = train()