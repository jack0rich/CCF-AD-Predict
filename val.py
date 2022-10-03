"""
Predict
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
from data_load import data_loader, read_labels, connect_da_la, set_id, PreDataset


def predict(model, dataset, f, len_=116):
    for i in range(len_):
        x = dataset[i][0].unsqueeze(1)
        id_ = dataset[i][1]
        out = model(x)
        result = out.data.max(1, keepdim=True)
        f.write(f"{id_},{result[1].item()}\n")
        print(f"{id_},{result[1].item()}")


if __name__ == "__main__":
    testa_path = "./test/testa.h5"
    testb_path = "./test/testb.h5"
    data_nums = 116
    testa_dataset = set_id(data_loader(testa_path, data_nums), "testa")
    testb_dataset = set_id(data_loader(testb_path, data_nums), "testb")
    testa_final = PreDataset(testa_dataset)
    testb_final = PreDataset(testb_dataset)
    ae = AE()
    ae.load_state_dict(torch.load("./AE_models/CnnAE3.pth"))
    model = ADNet(ae)
    model.load_state_dict(torch.load("./models/ADNet3.pth"))
    model.eval()

    with open("prediction2.csv", "a+") as file:
        file.write("test_id,label\n")
        predict(model, testa_final, file)
        predict(model, testb_final, file)






