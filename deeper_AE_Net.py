"""
deeper net
"""
import torch
from torch import nn
from torch.nn import Linear, Module, Sigmoid, ReLU, Conv3d, ConvTranspose3d, Sequential, BatchNorm3d, MaxPool3d


class AE2(Module):
    def __init__(self):
        super(AE2, self).__init__()
        self.encoder1 = Conv3d(1, 256, (5, 5, 5), 2, 0)
        self.bn1 = BatchNorm3d(256)
        self.encoder2 = Conv3d(256, 256, (3, 3, 3), 2, 0)
        self.bn2 = BatchNorm3d(256)

        self.decoder1 = ConvTranspose3d(256, 256, (5, 5, 5), 2, 0)
        self.bn3 = BatchNorm3d(256)
        self.decoder2 = ConvTranspose3d(256, 1, (3, 3, 3), 2, 0)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.encoder1(x)
        x = self.bn1(x)
        x = self.encoder2(x)
        x = self.bn2(x)
        x = self.decoder1(x)
        x = self.bn3(x)
        x = self.decoder2(x)
        return self.sigmoid(x)


class ADNet2(Module):
    def __init__(self, ae):
        super(ADNet2, self).__init__()
        self.conv1 = Conv3d(1, 256, (5, 5, 5), 3, 2)
        self.conv2 = Conv3d(256, 256, (3, 3, 3), 3, 1)
        self.conv1.weight = ae.encoder1.weight
        self.conv1.bias = ae.encoder1.bias
        self.conv2.weight = ae.encoder2.weight
        self.conv2.bias = ae.encoder2.bias
        self.maxP1 = nn.Sequential(
            MaxPool3d((5, 5, 5)),
            ReLU(),
            nn.BatchNorm3d(256)
        )
        self.act = nn.Sequential(
            ReLU(),
            nn.BatchNorm3d(256)
        )

        self.flatten = nn.Flatten()
        self.fc = Linear(2048, 128)
        self.out = Linear(128, 3)
        self.Softmax = LogSoftmax(dim=1)

        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxP1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.out(x)
        return self.Softmax(x)

