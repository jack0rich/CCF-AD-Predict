"""
Extract features
"""


import torch
from torch import nn
from torch.nn import Linear, Module, Sigmoid, ReLU, Conv3d, ConvTranspose3d, Sequential, BatchNorm3d, MaxPool3d

# 79*95*79


class AE(Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Conv3d(1, 256, (5, 5, 5), 2, 0)
        self.bn = BatchNorm3d(256)

        self.decoder = ConvTranspose3d(256, 1, (5, 5, 5), 2, 0)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.bn(x)
        x = self.decoder(x)
        return self.sigmoid(x)



