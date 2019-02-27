# -*- coding: UTF-8 -*-
import torch.nn as nn
import captcha_setting

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.avgpool = nn.AvgPool2d((captcha_setting.IMAGE_WIDTH//8))
        # self.fc = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.Dropout(0.5),  # drop 50% of the neuron
        #     nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(128, captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = self.fc2(out)
        return out

