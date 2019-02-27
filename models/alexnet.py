import torch
import torch.nn as nn
import torchvision.models as models
from models.capsule import CapConv2d


class AlexNet(nn.Module):

    def __init__(self, num_classes=365):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

#
# class FCView(nn.Module):
#     def __init__(self):
#         super(FCView, self).__init__()
#
#     def forward(self, x):
#         nB = x.data.size(0)
#         x = x.view(nB,-1)
#         return x
#     def __repr__(self):
#         return 'view(nB, -1)'
#


# class CaffeNet_David(nn.Module):
#     def __init__(self, dropout=True, bn=False, num_classes=365):
#         super(CaffeNet_David, self).__init__()
#         self.dropout = dropout
#         self.bn = bn
#         if self.bn:
#             self.bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
#             self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
#             self.bn3 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.9, affine=True)
#             self.bn4 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.9, affine=True)
#             self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
#             self.bn6 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.9, affine=True)
#             self.bn7 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.9, affine=True)
#         self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
#
#         self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
#
#         self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
#         self.relu4 = nn.ReLU(inplace=True)
#         self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
#         self.relu5 = nn.ReLU(inplace=True)
#         self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
#
#         if dropout:
#             self.drop1 = nn.Dropout()
#             self.drop2 = nn.Dropout()
#         self.fc6 = nn.Sequential(
#             FCView(),
#             nn.Linear(9216, 4096),
#         )
#         self.relu6 = nn.ReLU(inplace=True)
#         self.fc7 = nn.Linear(in_features=4096, out_features=4096)
#         self.relu7 = nn.ReLU(inplace=True)
#         self.fc8 = nn.Linear(in_features=4096, out_features=num_classes)
#
#     def forward(self, x):
#         if self.bn:
#             x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#             x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
#             x = self.relu5(self.bn5(self.conv5(self.relu4(self.bn4(self.conv4(self.relu3(self.bn3(self.conv3(x)))))))))
#             if self.dropout:
#                 x = self.fc8(self.relu7(self.bn7(self.fc7(self.drop2(self.relu6(self.bn6(self.fc6(self.drop1(self.pool5(x))))))))))
#             else:
#                 x = self.fc8(self.relu7(self.bn7(self.fc7(self.relu6(self.bn6(self.fc6(self.pool5(x))))))))
#         else:
#             x = self.pool1(self.relu1(self.conv1(x)))
#             x = self.pool2(self.relu2(self.conv2(x)))
#             x = self.relu5(self.conv5(self.relu4(self.conv4(self.relu3(self.conv3(x))))))
#             if self.dropout:
#                 x = self.fc8(self.relu7(self.fc7(self.drop2(self.relu6(self.fc6(self.drop1(self.pool5(x))))))))
#             else:
#                 x = self.fc8(self.relu7(self.fc7(self.relu6(self.fc6(self.pool5(x))))))
#         return x