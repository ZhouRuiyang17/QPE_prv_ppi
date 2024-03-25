import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import matplotlib.pyplot as plt
from tools import ml
import tqdm
from torch import nn
from data_loader import mini, maxi
from torch import optim
import data_loader
import numpy as np
import torch
from Src import models
from torch.utils import data


class WeightedMSELoss(nn.Module):
    def __init__(self, weights, edge):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights  # 传入的权重
        self.edge = edge

    def forward(self, predicted, target):
        edge = self.edge
        for i, _ in enumerate(edge[:-1]):
            loc = np.where((target.cpu().numpy() > edge[i]) & (target.cpu().numpy() <= edge[i + 1]))[0]
            if i == 0:
                loss = self.weights[i] * (predicted[loc] - target[loc]) ** 2
            else:
                loss = torch.cat([loss, self.weights[i] * (predicted[loc] - target[loc]) ** 2])
        loss = torch.sum(loss) / np.sum(weights)
        return loss


if __name__ == '__main__':

    batch_size = 64
    num_epochs = 200
    # device = 'cuda'
    device = 'cpu'
    save_loss_min = 100

    weights_path = './weights'
    model = models.CNN(num_classes=1).to(device)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    train_loss = open(weights_path + '/train_loss.txt', 'w')
    val_loss = open(weights_path + '/val_loss.txt', 'w')
    x_train, x_vali, x_test, y_train, y_vali, y_test = data_loader.get_data()

    dataset_train = data_loader.Dataset(x_train, y_train)
    dataset_val = data_loader.Dataset(x_vali, y_vali)

    train_loader = data.DataLoader(dataset_train, batch_size=batch_size,
                                   shuffle=True, num_workers=4)
    val_loader = data.DataLoader(dataset_val, batch_size=batch_size,
                                 shuffle=False, num_workers=4)
    # criterion = nn.MSELoss()
    edge = np.array(list(range(0, 52, 2)) + [100])
    weights = edge[:-1] + 1
    criterion = WeightedMSELoss(torch.tensor(weights), ml.min_max(edge, mini[-1], maxi[-1]))
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_steps = len(train_loader)
    train_loss_all, val_loss_all = [], []
    for epoch in range(num_epochs):
        print(epoch)
        dt_size = len(train_loader.dataset)
        dt_size_val = len(val_loader.dataset)
        epoch_loss = 0
# =============================================================================
#         pbar = tqdm.tqdm(
#             total=dt_size // batch_size,
#             desc=f'Epoch {epoch + 1} / {num_epochs}',
#             postfix=dict,
#             miniters=.3
#         )
# =============================================================================
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

# =============================================================================
#             pbar.set_postfix(**{
#                 'train_loss': epoch_loss / (i + 1),
#             })
#             pbar.update(1)
# =============================================================================
        train_loss.write(str(epoch_loss / i))
        train_loss.write('\n')
        train_loss_all.append(epoch_loss / i)
# =============================================================================
#         pbar.close()
#         pbar = tqdm.tqdm(
#             total=dt_size_val // batch_size,
#             desc=f'Val_Epoch {epoch + 1} / {num_epochs}',
#             postfix=dict,
#             miniters=.3
#         )
# =============================================================================
        epoch_loss_val = 0
        model.eval()
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
            epoch_loss_val += loss.item()
# =============================================================================
#             pbar.set_postfix(**{
#                 'val_loss': epoch_loss_val / (i + 1),
#             })
#             pbar.update(1)
#         pbar.close()
# =============================================================================
        val_loss.write(str(epoch_loss_val / i))
        val_loss.write('\n')
        val_loss_all.append(epoch_loss_val / i)

        if save_loss_min > epoch_loss_val / i:
            save_loss_min = epoch_loss_val / i
            torch.save(model.state_dict(), weights_path + '/weights.pth')
    print("训练完成！")
    plt.figure(figsize=(12, 12))
    plt.plot(train_loss_all, label='train loss')
    plt.plot(val_loss_all, label='vali loss')
    plt.legend()
    plt.savefig(weights_path + '/loss.png')