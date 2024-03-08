# ----other
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# ----torch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def spliter(x, y, ratio):
    x = x.copy()
    y = y.copy()
    test_size = ratio[-1] / sum(ratio)
    vali_size = ratio[1] / sum(ratio[:-1])

    from sklearn.model_selection import train_test_split
    x1, x_test, y1, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_vali, y_train, y_vali = train_test_split(x1, y1, test_size=vali_size)
    return [x_train, x_vali, x_test, y_train, y_vali, y_test]


# ----scaler
def min_max(data, mini, maxi):
    data[data < mini] = mini
    data[data > maxi] = maxi
    data_norm = (data - mini) / (maxi - mini)
    return data_norm


def min_max_rev(data_norm, mini, maxi):
    data = data_norm * (maxi - mini) + mini
    return data


# ----【1】【2】
def loader(x, y, batch_size=64):
    # ----【1】
    data = TensorDataset(x, y)
    # ----【2】
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=1)

    return dataloader


# ----【4】
# size：数据集长度
# batch：batch id，从0开始计数
# loss：是一个tensor，.item()用来调用内部的值，表示这个batch的平均loss，current
# current: 已经跑过的数据集， = (batch id + 1) * len(X) or batch_size
def train(dataloader, model, loss_fn, optimizer):
    xxx = []
    yyy = []
    ppp = []
    lll = []

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # =============================================================================
        #         if batch % 5 == 0:
        #             loss, current = loss.item(), (batch + 1) * len(X)
        #             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # =============================================================================

        xxx += X.tolist()
        yyy += y.tolist()
        ppp += pred.tolist()
        lll += [loss_fn(pred, y).item()]
    return xxx, yyy, ppp, sum(lll) / len(lll)


# test_loss：总的loss，通过除以num_batches得到平均loss
# correct：这里指的是分类问题的正确数，通过除以size得到正确率
def test(dataloader, model, loss_fn):
    xxx = []
    yyy = []
    ppp = []
    lll = []

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            xxx += X.tolist()
            yyy += y.tolist()
            ppp += pred.tolist()
            lll += [loss_fn(pred, y).item()]
    test_loss /= num_batches
    correct /= size
    # =============================================================================
    #     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # =============================================================================
    return xxx, yyy, ppp, sum(lll) / len(lll)


class Net(nn.Module):
    # def __init__(self,n_input,n_hidden,n_output):
    def __init__(self):
        super(Net, self).__init__()
        # self.hidden1 = nn.Linear(n_input,n_hidden)
        # self.hidden2 = nn.Linear(n_hidden,n_hidden)
        # self.predict = nn.Linear(n_hidden,n_output)

        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
        )

    def forward(self, input):
        # out = self.hidden1(input)
        # out = F.relu(out)
        # out = self.hidden2(out)
        # out = F.sigmoid(out)
        # out =self.predict(out)

        out = self.net(input)
        return out

