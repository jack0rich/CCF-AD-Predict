"""
Base 3D-CNN
"""


import torch
from torch import nn
from _3D_AE import AE
from torch.nn.functional import relu
from torch.nn import Linear, Module, Sigmoid, ReLU, Conv3d, MaxPool3d, AdaptiveAvgPool3d, BatchNorm3d, LogSoftmax


class LeNet_3D(Module):
    def __init__(self):
        super(LeNet_3D, self).__init__()
        self.conv1 = Conv3d(1, 6, (2, 2, 2), stride=1)
        self.conv2 = Conv3d(6, 16, (3, 3, 3), stride=1)

        self.maxP = MaxPool3d(2, 2)
        self.avgP = AdaptiveAvgPool3d((1, 32, 32))

        self.fc1 = Linear(1024*16, 256)
        self.fc2 = Linear(256, 64)
        self.out = Linear(64, 3)

        self.dropout = nn.Dropout()

        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = relu(self.maxP(self.conv1(x)))
        x = relu(self.avgP(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        x = self.out(x)
        print(x.size())
        return self.logSoftmax(x)


class C3DNet(Module):
    def __init__(self):
        super(C3DNet, self).__init__()

        self.conv1 = Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1)
        self.maxP1 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.maxP2 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3b = Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.maxP3 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv4b = Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.maxP4 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.conv5b = Conv3d(512, 512, kernel_size=(3, 3, 3), padding=1)
        self.maxP5 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc1 = Linear(18432, 512)
        self.fc2 = Linear(512, 256)
        self.out = Linear(256, 3)

        self.dropout = nn.Dropout()

        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.maxP1(x)

        x = relu(self.conv2(x))
        x = self.maxP2(x)

        x = relu(self.conv3a(x))
        x = relu(self.conv3b(x))
        x = self.maxP3(x)

        x = relu(self.conv4a(x))
        x = relu(self.conv4b(x))
        x = self.maxP4(x)

        x = relu(self.conv5a(x))
        x = relu(self.conv5b(x))
        x = self.maxP5(x)

        x = torch.flatten(x, 1)

        x = relu(self.fc1(x))
        x = self.dropout(x)

        x = relu(self.fc2(x))
        x = self.dropout(x)

        x = self.out(x)
        return self.logSoftmax(x)


class ADNet(Module):
    def __init__(self, ae):
        super(ADNet, self).__init__()
        self.conv = Conv3d(1, 256, (5, 5, 5), 3, 2)
        self.conv.weight = ae.encoder.weight
        self.conv.bias = ae.encoder.bias
        self.maxP = nn.Sequential(
            MaxPool3d((5, 5, 5)),
            ReLU(),
            nn.BatchNorm3d(256)
        )

        self.flatten = nn.Flatten()
        self.fc = Linear(38400, 128)
        self.out = Linear(128, 3)
        self.Softmax = LogSoftmax(dim=1)

        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv(x)
        x = self.maxP(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.out(x)
        return self.Softmax(x)




