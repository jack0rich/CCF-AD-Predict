"""
DATA LOAD
"""

import csv
import h5py
import numpy
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize


def data_loader(path="./train/train_pre_data.h5", len_=300):
    data_list = []
    with h5py.File(path, 'r') as F:
        dataset = F['data']
        for i in range(len_):
            data_list.append(torch.tensor(dataset[i]))
        return data_list


def read_labels(path="./train/train_pre_label.csv"):
    with open(path, 'r') as file:
        loaded = csv.DictReader(file)
        labels = []
        for i in loaded:
            labels.append(i)
        return labels


def connect_da_la(datas, labels_):
    dataset = []
    for i in range(300):
        label = labels_[i]['label']
        if label == '0':
            label = torch.tensor(0)  # [1.0, 0.0, 0.0]
        elif label == '1':
            label = torch.tensor(1)  # [0.0, 1.0, 0.0]
        elif label == '2':
            label = torch.tensor(2)  # [0.0, 0.0, 1.0]
        else:
            print("ERROR", i)
            return
        dataset.append((datas[i], label))
    return dataset


def k_fold(data_array, k_size=5):
    result = []
    k = int(len(data_array)/k_size)
    for i in range(k_size):
        result.append(data_array[i*k:i*k+k])
    return result


def set_id(datas, id_, len_=116):
    result = []
    for i in range(len_):
        result.append((datas[i], f"{id_}_{i}"))
    return result


class MRIDataset(Dataset):
    def __init__(self, ori_data):
        self.ori_data = ori_data

    def __len__(self):
        return len(self.ori_data)

    def __getitem__(self, item):
        mean, std = self.ori_data[item][0].mean(), self.ori_data[item][0].std()
        mri_data = normalize(self.ori_data[item][0], mean, std)
        label = self.ori_data[item][1]
        return mri_data, label


class PreDataset(Dataset):
    def __init__(self, ori_data):
        self.ori_data = ori_data

    def __len__(self):
        return len(self.ori_data)

    def __getitem__(self, item):
        mean, std = self.ori_data[item][0].mean(), self.ori_data[item][0].std()
        nor_data = normalize(self.ori_data[item][0], mean, std)
        return nor_data, self.ori_data[item][1]



