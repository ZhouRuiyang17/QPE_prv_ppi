import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime
import my.ml as ml

import torch
from model import *

path = r'E:\QPE_prv_ppi_2_99\dataset聚合\20231221'
path_save = r'E:\QPE_prv_ppi_2_99\model\{}'.format(20231221)
if not os.path.exists(path_save):
    os.makedirs(path_save)
maxi = [75, 7, 14, 1, 100]
mini = [ 0, 0,  0, 0,   0]

def scaler(datas):
    for i, data in enumerate(datas):
        datas[i] = ml.min_max(data, mini[i], maxi[i])
    return datas
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
    
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    vali_x = torch.from_numpy(vali_x)
    vali_y = torch.from_numpy(vali_y)
    
    # [1][2]
    train = ml.loader(train_x, train_y)
    vali = ml.loader(vali_x, vali_y)

    # [3]
    net = CNN()
    optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)
    loss_func = torch.nn.MSELoss()
    
    # [5]
    plt.ion()
    plt.show()
    epochs = 100; loss = []; loss2 = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        aaaaa = ml.train(train, net, loss_func, optimizer)
        bbbbb = ml.test(vali, net, loss_func)
        
        loss += [aaaaa[-1]]
        loss2 += [bbbbb[-1]]
        if t % 10 == 0:
            plt.cla()
            rainrate = np.array(aaaaa[1]).flatten(); prediction = np.array(aaaaa[2]).flatten()
            plt.hist2d(rainrate, prediction,bins = np.arange(0,2,0.01), norm = colors.LogNorm())
            plt.xlim(0,2);plt.ylim(0,2);plt.plot([0,100],[0,100])
            plt.title('training')
            plt.pause(0.5)

            plt.cla()
            rainrate = np.array(bbbbb[1]).flatten(); prediction = np.array(bbbbb[2]).flatten()
            plt.hist2d(rainrate, prediction,bins = np.arange(0,2,0.01), norm = colors.LogNorm())
            plt.xlim(0,2);plt.ylim(0,2);plt.plot([0,100],[0,100])
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