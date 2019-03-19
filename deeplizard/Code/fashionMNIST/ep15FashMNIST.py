import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                              train=True, download=True,
                                              transform=transforms.Compose(transforms.ToTensor()))
test_set = torchvision.datasets.FashionMNIST(root='./data/test/FashionMNIST',
                                             train=False, download=True,
                                             transform=transforms.Compose(transforms.ToTensor()))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)

#batch = next(iter(train_loader))

#images, labels = batch  # features and labels
# images.shape = [10,1,28,28]
# labels.shape = [10]


IMAGE_SIZE = 28
CHANNEL = 1

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

class Network(nn.Module()):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self,t):
        # Max pooling over a (2, 2) window
        t = F.max_pool2d(F.relu(self.conv1(t)),(2,2))
        # If the size is a square you can only specify a single number
        t = F.max_pool2d(F.relu(self.conv2(t),2))
        t = t.view(-1,self.num_flat_features(t))
        t = F.relu(self.fc1)
        t = F.relu(self.fc2)
        t = F.relu(self.out)

        return t

def train():
    model = Network()
    opti = torch.optim.Adam(model.parameters(),lr = LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1,Train_epoch+1):
        for batchID, (label,image) in enumerate(train_loader):
            label,image = label.to(device), image.to(device)
            output = model(image)
            loss = criterion(output,label)
            opti.zero_grad()
            loss.backward()
            opti.step()
            if batchID % 1000 == 0:
                print('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, Train_epoch))
    return model

def test(model):
    with torch.no_grad():
        correct = 0
        total = 0
        for label, image in test_loader:
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            predicted = torch.argmax(outputs,dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    if __name__ == '__main__':
        model = train()
        test(model)






    # def num_flat_features(self,t):
    #     size = t.size()[1:] # all dimensions except the batch dimension
    #     numFeatures = 1
    #     for s in size:
    #         numFeatures *= s
    #     return numFeatures