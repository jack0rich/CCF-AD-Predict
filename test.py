"""
Test
"""


import h5py
import time
import torch
import numpy
from torch import nn
from tqdm import tqdm
from _3D_AE import AE
from torch.optim import Adam
from cnn_3d import C3DNet, ADNet
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize
from data_load import data_loader, read_labels, connect_da_la, set_id, MRIDataset


def predict(model, dataset, len_=300):
    connect = 0
    for i in range(len_):
        x = dataset[i][0].unsqueeze(1)
        label = dataset[i][1]
        out = model(x)
        result = out.data.max(1, keepdim=True)
        if label.item() == result[1].item():
            connect += 1
    print(connect/300)


if __name__ == "__main__":

    ori_dataset = data_loader()
    labels = read_labels()
    final_dataset = connect_da_la(ori_dataset, read_labels())
    final_dataset = MRIDataset(final_dataset)

    ae = AE()
    ae.load_state_dict(torch.load("./AE_models/CnnAE2.pth"))
    model = ADNet(ae)
    model.load_state_dict(torch.load("./models/ADNet2.pth"))
    model.eval()

    predict(model, final_dataset)

