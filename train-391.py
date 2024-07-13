import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import my.utils as utils

path_save = './model/based_on_202407/{}'.format('240713-cnn-9prv-wmse')
if not os.path.exists(path_save):
    os.makedirs(path_save)

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备:", device)

# input(f'res will be stored in:\n{path_save}\nshall we go on[y/n]?')

def plot(res1, res2, loss_train, loss_vali):
    t = len(loss_train)

    plt.cla()
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot()
    rainrate = np.array(res1[1]).flatten(); prediction = np.array(res1[2]).flatten()
    ax.hist2d(rainrate, prediction,bins = [np.arange(0,1,0.01)]*2, norm = colors.LogNorm())
    ax.plot([0,1],[0,1])
    ax.set_title('train')
    ax.set_aspect('equal')
    bias = np.sum(prediction - rainrate) / np.sum(rainrate)
    ax.text(0.8,0.9,bias)
    plt.savefig(path_save + '/train_epoch{}.png'.format(t), bbox_inches = 'tight')
    plt.close()

    plt.cla()
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot()
    rainrate = np.array(res2[1]).flatten(); prediction = np.array(res2[2]).flatten()
    ax.hist2d(rainrate, prediction,bins = [np.arange(0,1,0.01)]*2, norm = colors.LogNorm())
    ax.plot([0,1],[0,1])
    ax.set_title('vali')
    ax.set_aspect('equal')
    bias = np.sum(prediction - rainrate) / np.sum(rainrate)
    ax.text(0.8,0.9,bias)
    plt.savefig(path_save + '/vali_epoch{}.png'.format(t), bbox_inches = 'tight')
    plt.close()

    plt.cla()
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot()
    ax.plot(loss_train, label = 'train loss')
    ax.plot(loss_vali, label = 'vali loss')
    ax.set_aspect('auto')
    plt.legend()
    plt.savefig(path_save + '/loss.png', bbox_inches = 'tight')
    plt.close()


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, inputs, targets):
        # 定义目标值区间和对应的权重
        weight_intervals = [
            (0.0, 0.1, 1),
            (0.1, 0.2, 2),
            (0.2, 0.3, 3),
            (0.3, 0.4, 4),
            (0.4, 0.5, 5),
            (0.5, 0.6, 6),
            (0.6, 0.7, 7),
            (0.7, 0.8, 8),
            (0.8, 0.9, 9),
            (0.9, 1.0, 10)
        ]
        
        # 创建权重矩阵
        weights = torch.zeros_like(targets)
        for lower, upper, weight in weight_intervals:
            mask = (targets >= lower) & (targets < upper)
            weights[mask] = weight
        
        # 计算加权均方误差
        mse_loss = nn.MSELoss(reduction='none')  # 不做平均
        loss = mse_loss(inputs, targets)
        weighted_loss = loss * weights
        
        return weighted_loss.mean()  # 可以根据需要选择如何汇总损失

if __name__ == "__main__":
    
    # ----封装
    dataset_train = np.load('../dataset-3-9.npz')
    train_x = dataset_train['x_train'].astype(np.float32)
    train_y = dataset_train['y_train'].astype(np.float32)
    vali_x = dataset_train['x_vali'].astype(np.float32)
    vali_y = dataset_train['y_vali'].astype(np.float32)
    print('Data loaded')

    '''裁剪数据和归一化'''
    train_x[:,[0,3,6]] = utils.scaler(train_x[:,[0,3,6]], 'ref')
    vali_x[:, [0,3,6]] = utils.scaler(vali_x[:, [0,3,6]], 'ref')
    train_x[:,[1,4,7]] = utils.scaler(train_x[:,[1,4,7]], 'zdr')
    vali_x[:, [1,4,7]] = utils.scaler(vali_x[:, [1,4,7]], 'zdr')
    train_x[:,[2,5,8]] = utils.scaler(train_x[:,[2,5,8]], 'kdp')
    vali_x[:, [2,5,8]] = utils.scaler(vali_x[:, [2,5,8]], 'kdp')

    train_y = train_y[:, 0].reshape(-1, 1)
    vali_y = vali_y[:, 0].reshape(-1, 1)
    train_y = utils.scaler(train_y, 'rr')
    vali_y = utils.scaler(vali_y, 'rr')
    # train_y = utils.scaler(np.log10(train_y), 'log10rr')
    # vali_y = utils.scaler(np.log10(vali_y), 'log10rr')

    '''数据加载'''
    train = utils.loader(train_x, train_y, device, 64)
    vali = utils.loader(vali_x, vali_y, device, 64)

    
    '''训练'''
    model = CNN(9,1).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4, weight_decay = 1e-4)
    # loss_func = torch.nn.MSELoss()
    loss_func = WeightedMSELoss()
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)


    epochs = 500
    loss_train = []; loss_vali = []
    params = []; slopes = []; positive_counter = 0; positive_position = []
    for t in range(epochs):
        print(f"-------------------------------\nEpoch {t+1}")
        
        res1, res2 = utils.trainer(train, vali, model, loss_func, optimizer)
        scheduler.step()
        print(f"Epoch {t + 1}, Learning Rate:")
        for param_group in optimizer.param_groups:
            print(param_group['lr']) # 打印更新后的学习率
        
        loss_train += [res1[-1]]
        loss_vali += [res2[-1]]
        if t % 10 == 0 or t == 5:           
            plot(res1, res2, loss_train, loss_vali)
        
        '''
        always store the LAST epochs/10 of params and save the LAST params
        check the slope of the loss_vali of the LAST epochs/10 
        when slope > 0, count, and record the SLOPE and POSITION
        when counter == 20, stop
        '''
        if len(params) < epochs/10:
            params.append(model.state_dict())
        else:
            params.append(model.state_dict())
            params = params[1:]
            flag, slope = utils.early_stop(loss_vali, int(epochs/10))
            torch.save(params[-1], path_save + '/' + "cnn.pth")

            if flag and t >= 50:
                slopes += [slope]
                positive_position += [t]
                positive_counter += flag
            if positive_counter == 50:
                torch.save(params[-1], path_save + '/' + "cnn.pth")
                print('early stop at epoch:{}'.format(t))
                plot(res1, res2, loss_train, loss_vali)
                print(slopes)
                print(positive_position)
                break

            # flag_stop = utils.early_stop_ptrend(loss_vali, 10)
            # if flag_stop:
            #     torch.save(params[-1], path_save + '/' + "cnn.pth")
            #     print('early stop at epoch:{}'.format(t))
            #     plot(res1, res2, loss_train, loss_vali)
            #     break
    
    print("Done!")

    
    
    # if positive_counter != 20:
    #     torch.save(model.state_dict(), path_save + '/' + "cnn.pth")
    #     print('finish all epochs:{}'.format(epochs))  

    loss_arr = np.array([loss_train, loss_vali]).T
    np.savetxt(f'{path_save}/loss.csv', loss_arr, delimiter=',')


    
