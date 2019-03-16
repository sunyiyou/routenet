import torch
import torch.nn as nn
import torch.nn.functional as F
from models.capsule import CapConv2d

class LeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class ResLeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.fc1   = nn.Linear(32, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.avg_pool2d(out, 5)
        out = self.fc1(out.squeeze())
        return out

class CapLeNet(nn.Module):
    def __init__(self,in_channels=3, num_classes=10):
        super(CapLeNet, self).__init__()
        self.conv1 = CapConv2d(in_channels, 3, 5, out_groups=2)
        self.conv2 = CapConv2d(3, 4, 5, in_groups=2, out_groups=4)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet2(nn.Module):
    def __init__(self,in_channels=3, num_classes=10):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = nn.Conv2d(128, 512, 3, 1, padding=1)
        self.bn4 = nn.BatchNorm2d(512, affine=False)
        self.conv5 = nn.Conv2d(512, 512, 3, 1, padding=1)

        self.fc1   = nn.Linear(512, 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn1(out)
        out = F.max_pool2d(out, 2, 2)

        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = F.max_pool2d(out, 2, 2)

        out = F.relu(self.conv3(out))
        out = self.bn3(out)
        out = F.max_pool2d(out, 2, 2)

        out = F.relu(self.conv4(out))
        out = self.bn4(out)
        out = F.max_pool2d(out, 2, 2)

        out = F.relu(self.conv5(out))
        out = F.max_pool2d(out, 2, 2)


        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CapLeNet2(nn.Module):
    def __init__(self,in_channels=3, num_classes=10):
        super(CapLeNet2, self).__init__()
        self.conv1 = CapConv2d(in_channels, 64, 3, padding=1, group=1)
        self.gn1 = nn.GroupNorm(16, 64, affine=False)
        self.conv2 = CapConv2d(64, 64, 3, 1, padding=1, group=16)
        self.gn2 = nn.GroupNorm(16, 64, affine=False)
        self.conv3 = CapConv2d(64, 128, 3, 1, padding=1, group=16)
        self.gn3 = nn.GroupNorm(32, 128, affine=False)
        self.conv4 = CapConv2d(128, 512, 3, 1, padding=1, group=32)
        self.gn4 = nn.GroupNorm(32, 512, affine=False)
        self.conv5 = CapConv2d(512, 512, 3, 1, padding=1, group=32)
        self.gn5 = nn.GroupNorm(32, 512, affine=False)
        self.conv6 = CapConv2d(512, 512, 3, 1, padding=1, group=32)

        self.fc1   = nn.Linear(512, 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.gn1(out)
        out = F.max_pool2d(out, 2, 2)

        out = F.relu(self.conv2(out))
        out = self.gn2(out)
        out = F.max_pool2d(out, 2, 2)

        out = F.relu(self.conv3(out))
        out = self.gn3(out)
        out = F.max_pool2d(out, 2, 2)

        out = F.relu(self.conv4(out))
        out = self.gn4(out)
        out = F.max_pool2d(out, 2, 2)

        out = F.relu(self.conv5(out))
        out = self.gn5(out)
        out = F.relu(self.conv6(out))
        out = F.max_pool2d(out, 2, 2)


        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
