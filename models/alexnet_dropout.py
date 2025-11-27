import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self,num_classes=200):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1,padding=2)
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,1,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,1,padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        # drop out layers
        self.drop = nn.Dropout()

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,3,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3,2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 3,2)
        x = torch.flatten(x,1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
