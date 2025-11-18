import torch
from torch import nn


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 10)
    
    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc1(x)
        return x

class CNN(nn.Module): # shape CIFAR10 : 32*32*3
    '''Modify the script models.py to complete the class CNN. The class should
contain 3 convolutional layers with ReLU activation and Max pooling and 2 fully
connected layers. The number of filters in the convolutional layers should be 32, 64,
and 128. The size of the fully connected layers should be 512 and 10.'''
    def __init__(self):
        super(CNN, self).__init__()
        ### To do 4
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, 10)

        self.pool = nn.MaxPool2d(2, 2)  # utilisé 3 fois
        self.activation = nn.ReLU()

    def forward(self, x):
        ### To do 4
       # Convolution block 1
        x = self.activation((self.conv1(x)))
        x = self.pool(x)

        # Convolution block 2
        x = self.activation((self.conv2(x)))
        x = self.pool(x)

        # Convolution block 3
        x = self.activation((self.conv3(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 128*4*4)

        # Fully connected
        x = self.activation((self.fc1(x)))
        x = self.fc2(x)

        return x



class ResNet18(nn.Module):
    def __init__(self):
        """To Do 7. Modify the class ResNet18 in the script models.py to fine-tune to replace
the last layer of the ResNet18 model (self.resnet.fc) with a linear layer with 512
input features and 10 output features."""
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        ## To do 7
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)



