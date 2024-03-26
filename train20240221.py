import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime
import my.ml as ml

import torch
from model import *
import matplotlib.colors as colors
import torch.nn as nn

path = r'D:\data\dataset\prv_ppi\dataset20240101\20240221'
path_save = r'E:\QPE_prv_ppi_2_99\model\20240101-20240221\{}'.format(20240221)
if not os.path.exists(path_save):
    os.makedirs(path_save)
maxi = [70, 7, 7, 1, 100]
mini = [ 0, 0,  0, 0,   0]

def scaler(datas):
    for i, data in enumerate(datas):
        datas[i] = ml.min_max(data, mini[i], maxi[i])
    return datas

class WeightedMSELoss(nn.Module):
    def __init__(self, weights, edge):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights  # 传入的权重
        self.edge = edge

    def forward(self, predicted, target):
        edge = self.edge
        for i, _ in enumerate(edge[:-1]):
            loc = np.where((target > edge[i]) & (target <= edge[i+1]))[0]
            if i == 0:
                loss = self.weights[i] * (predicted[loc] - target[loc])**2
            else:
                loss = torch.cat([loss, self.weights[i] * (predicted[loc] - target[loc])**2])
        # 计算每个样本的损失并加权求和
        loss = torch.sum(loss) / np.sum(weights)
        # loss = torch.mean((predicted - target)**2)
        return loss

class WeightedMSELoss_ver2(nn.Module):
    def __init__(self):
        super(WeightedMSELoss_ver2, self).__init__()


    def forward(self, predicted, target):
        loss = []
        for i in range(len(target)):
            loss += [target[i]*100 * (predicted[i] - target[i])**2]
        # 计算每个样本的损失并加权求和
        loss = torch.Tensor(loss)
        loss = (torch.mean(loss))
        loss2 = torch.mean(target*100*(predicted - target)**2)
        return loss2


if __name__ == "__main__":
    
    train_x = np.load(os.path.join(path,'train_x.npy'), allow_pickle=True).astype(np.float32)
    train_y = np.load(os.path.join(path,'train_y.npy'), allow_pickle=True).reshape(-1, 1).astype(np.float32)
    vali_x = np.load(os.path.join(path,'vali_x.npy'), allow_pickle=True).astype(np.float32)
    vali_y = np.load(os.path.join(path,'vali_y.npy'), allow_pickle=True).reshape(-1, 1).astype(np.float32)
    test_x = np.load(os.path.join(path,'test_x.npy'), allow_pickle=True).astype(np.float32)
    test_y = np.load(os.path.join(path,'test_y.npy'), allow_pickle=True).astype(np.float32)
    
    _ls = scaler([train_x[:, 0:2], train_x[:, 2:4], train_x[:, 4:6], train_x[:, 6:8], train_y])
    train_x[:, 0:2], train_x[:, 2:4], train_x[:, 4:6], train_x[:, 6:8], train_y = _ls
    
    _ls = scaler([vali_x[:, 0:2], vali_x[:, 2:4], vali_x[:, 4:6], vali_x[:, 6:8], vali_y])
    vali_x[:, 0:2], vali_x[:, 2:4], vali_x[:, 4:6], vali_x[:, 6:8], vali_y = _ls
    
    _ls = scaler([test_x[:, 0:2], test_x[:, 2:4], test_x[:, 4:6], test_x[:, 6:8]])
    test_x[:, 0:2], test_x[:, 2:4], test_x[:, 4:6], test_x[:, 6:8] = _ls
    
    train_x = torch.from_numpy(train_x[:, :6])
    train_y = torch.from_numpy(train_y)
    vali_x = torch.from_numpy(vali_x[:, :6])
    vali_y = torch.from_numpy(vali_y)
    test_x = torch.from_numpy(test_x[:, :6])
    
    # [1][2]
    train = ml.loader(train_x, train_y, 32)
    vali = ml.loader(vali_x, vali_y, 32)
    
    # [3]
    net = CNN()
    optimizer = torch.optim.Adam(net.parameters(),lr = 1e-3, weight_decay = 1e-4)
    # loss_func = torch.nn.MSELoss()
    # weights = np.load(path + '\\' + 'weights_1225.npy')
    # edge = np.load(path + '\\' + 'edge.npy')
    # '''2023.12.26重写权重'''
    edge = np.array(list(range(0, 52, 2)) + [100])
    weights = edge[:-1]+1
    loss_func = WeightedMSELoss(torch.tensor(weights), ml.min_max(edge, mini[-1], maxi[-1]))
    # loss_func = WeightedMSELoss_ver2()
    
    # [5]
    plt.ion()
    plt.show()
    epochs = 500; loss = []; loss2 = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        aaaaa = ml.train(train, net, loss_func, optimizer)
        bbbbb = ml.test(vali, net, loss_func)
        
        loss += [aaaaa[-1]]
        loss2 += [bbbbb[-1]]
        if t % 50 == 0:
            plt.cla()
            rainrate = np.array(aaaaa[1]).flatten(); prediction = np.array(aaaaa[2]).flatten()
            plt.hist2d(rainrate, prediction,bins = 100, norm = colors.LogNorm())
            # plt.scatter(rainrate, prediction)
            # plt.xlim(0,2);plt.ylim(0,2);
            plt.plot([0,1],[0,1])
            plt.title('training')
            plt.pause(0.5)

            plt.cla()
            rainrate = np.array(bbbbb[1]).flatten(); prediction = np.array(bbbbb[2]).flatten()
            plt.hist2d(rainrate, prediction,bins = 100, norm = colors.LogNorm())
            # plt.scatter(rainrate, prediction)
            # plt.xlim(0,2);plt.ylim(0,2);
            plt.plot([0,1],[0,1])
            plt.title('validating')
            plt.pause(0.5)

            plt.cla()
            plt.plot(loss, label = 'train loss')
            plt.plot(loss2, label = 'vali loss')
            plt.legend()
            plt.pause(0.5)
    print("Done!")
    plt.ioff()
    plt.show()
    
    # [6]
    torch.save(net.state_dict(), path_save + '\\' + "cnn.pth")
    
    #%%
    # [7]
    net = CNN()
    net.load_state_dict(torch.load(path_save + '\\' + "cnn.pth"))
    net.eval()
    with torch.no_grad():
        pred = net(test_x)
    
    pred = pred.view(-1).detach().numpy()
    # pred = np.log(pred + 1)
    pred = ml.min_max_rev(pred, mini[-1], maxi[-1])
    # pred = 10**pred
    
    # metrics = []
    # scatter = mytools.Scatter(y_test, zr300)
    # scatter.plot3(bins = [np.arange(100), np.arange(100)], lim=[[0,100],[0,100]],draw_line = 1)
    # metrics += [scatter.evaluate().copy()]
    # df = pd.DataFrame(metrics)
    
    metrics_ml = {}
    # metrics_300 = {}

    # ----评估
    scatter = mt.Scatter(test_y, pred)
    metrics_ml['all'] = scatter.evaluate().copy()
    scatter.plot3(bins = [np.arange(100), np.arange(100)], lim=[[0.1,100],[0.1,100]],draw_line = 1,
                  show_metrics=1, label = ['rain rate (gauge) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'ML')
    
    # scatter = mytools.Scatter(y_test, zr300)
    # metrics_300['all'] = scatter.evaluate().copy()
    # scatter.plot3(bins = [np.arange(100), np.arange(100)], lim=[[1,100],[1,100]],draw_line = 1,
    #               show_metrics=1, label = ['rain rate (gauge) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'Z-R relation')

    
    # ----分段评估
    # edge = [0.1, 10, 20, 30, 40, 50, 100, 200]
    # for i in range(len(edge) - 1):     
    #     loc = np.where((y_test >= edge[i]) & (y_test < edge[i+1]))
        
    #     scatter = mytools.Scatter(y_test[loc], pred[loc])
    #     metrics_ml['{}-{}'.format(str(edge[i]), str(edge[i+1]))] = scatter.evaluate().copy()

        # scatter = mytools.Scatter(y_test[loc], zr300[loc])
        # metrics_300['{}-{}'.format(str(edge[i]), str(edge[i+1]))] = scatter.evaluate().copy()
       
    # metrics_ml = pd.DataFrame(metrics_ml)
    # metrics_300 = pd.DataFrame(metrics_300)
    # metrics = pd.concat([metrics_ml, metrics_300], axis=0)

    # metrics.to_excel( os.path.join(path_save, 'stat.xlsx'))
    # metrics_ml.to_excel( os.path.join(path_save, 'statmlp.xlsx'))
    # metrics_300.to_excel( os.path.join(path_save, 'stat300.xlsx'))