import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                              train=True,download=True,
                                              transform=transforms.Compose(transforms.ToTensor()))
train_loader = torch.utils.data.DataLoader(train_set,batch_size=10)

batch = next(iter(train_loader))

images, labels = batch #features and labels
#images.shape = [10,1,28,28]
#labels.shape = [10]

class Network(nn.Module()):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self,t):

        return t