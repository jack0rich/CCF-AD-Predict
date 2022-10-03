"""
Make AE work
"""


import time
import torch
from torch import nn
from tqdm import tqdm
from _3D_AE import AE
from torch.optim import Adam
from cnn_3d import LeNet_3D, C3DNet
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize
from data_load import data_loader, read_labels, connect_da_la, MRIDataset


def matplot_loss(train_loss):
    plt.plot(train_loss, label="train_loss")
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel("times")
    plt.title("loss")
    plt.show()


def train(data, model, opt, loss_list, dev='cpu'):
    start = time.time()
    data = iter(data)
    for i in tqdm(range(19), desc="训练进度", ncols=100):
        opt.zero_grad()
        x, label = next(data)
        x = x.to(dev)
        out = model(x)
        loss = nn.functional.binary_cross_entropy(out, torch.sigmoid(x))
        loss.backward()
        if i % 10 == 0:
            loss_list.append(loss.item())
        opt.step()
    end = time.time()
    print(f"#训练时长:{round(((end - start) / 60), 2)}分钟")


if __name__ == "__main__":
    ori_dataset = data_loader()
    labels = read_labels()
    final_dataset = connect_da_la(ori_dataset, read_labels())
    final_dataset = MRIDataset(final_dataset)
    final_dataset = DataLoader(final_dataset, batch_size=16, shuffle=True)

    device = 'cpu'
    model = AE().to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    losses = []

    for epoch in range(25):
        train(final_dataset, model, optimizer, losses)
        time.sleep(0.1)
        if epoch % 10 == 0 and epoch != 0:
            print("====================")
            print(f"{epoch} done | sleep {30+0.5*epoch}s")
            time.sleep(30+0.5*epoch)

    torch.save(model.state_dict(), "./AE_models/CnnAE3.pth")
    matplot_loss(losses)



