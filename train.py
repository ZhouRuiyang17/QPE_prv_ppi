import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import my.utils.utils_ml as utils
from common_tool import *

import logging

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, inputs, targets):
        rr_target = targets[:, 0]

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
            mask = (rr_target >= lower) & (rr_target < upper)
            weights[mask] = weight
        # weights = weights.unsqueeze(1).expand(-1, 3)
        # logging.info(weights[:100])

        # 计算加权均方误差
        mse_loss = nn.MSELoss(reduction='none')  # 不做平均
        loss = mse_loss(inputs, targets)
        weighted_loss = loss * weights
        
        return weighted_loss.mean()  # 可以根据需要选择如何汇总损失

def process_for_train_999_3(device):
    '''加载:(ele1, ele2, ele3)'''
    dataset_train = np.load('../dataset-3-9.npz')
    train_x = dataset_train['x_train'].astype(np.float32)
    train_y = dataset_train['y_train'].astype(np.float32)
    vali_x = dataset_train['x_vali'].astype(np.float32)
    vali_y = dataset_train['y_vali'].astype(np.float32)



    '''裁剪数据和归一化'''
    train_x[:,[0,3,6]] = utils.scaler(train_x[:,[0,3,6]], 'ref')
    vali_x[:, [0,3,6]] = utils.scaler(vali_x[:, [0,3,6]], 'ref')
    train_x[:,[1,4,7]] = utils.scaler(train_x[:,[1,4,7]], 'zdr')
    vali_x[:, [1,4,7]] = utils.scaler(vali_x[:, [1,4,7]], 'zdr')
    train_x[:,[2,5,8]] = utils.scaler(train_x[:,[2,5,8]], 'kdp')
    vali_x[:, [2,5,8]] = utils.scaler(vali_x[:, [2,5,8]], 'kdp')

    train_y = train_y[:, :-1]
    vali_y = vali_y[:, :-1]
    train_y[:,0] = utils.scaler(train_y[:,0], 'rr')
    vali_y[:,0] = utils.scaler(vali_y[:,0], 'rr')
    train_y[:,1] = utils.scaler(train_y[:,1], 'D0')
    vali_y[:,1] = utils.scaler(vali_y[:,1], 'D0')
    train_y[:,2] = utils.scaler(train_y[:,2], 'log10Nw')
    vali_y[:,2] = utils.scaler(vali_y[:,2], 'log10Nw')



    '''数据加载'''
    train = utils.loader(train_x, train_y, device, 64)
    vali = utils.loader(vali_x, vali_y, device, 64)

    return train, vali
def process_for_train_999_1(device):
    '''加载:(ele1, ele2, ele3)'''
    dataset_train = np.load('../dataset-3-9.npz')
    train_x = dataset_train['x_train'].astype(np.float32)
    train_y = dataset_train['y_train'].astype(np.float32)
    vali_x = dataset_train['x_vali'].astype(np.float32)
    vali_y = dataset_train['y_vali'].astype(np.float32)



    '''裁剪数据和归一化'''
    train_x[:,[0,3,6]] = utils.scaler(train_x[:,[0,3,6]], 'ref')
    vali_x[:, [0,3,6]] = utils.scaler(vali_x[:, [0,3,6]], 'ref')
    train_x[:,[1,4,7]] = utils.scaler(train_x[:,[1,4,7]], 'zdr')
    vali_x[:, [1,4,7]] = utils.scaler(vali_x[:, [1,4,7]], 'zdr')
    train_x[:,[2,5,8]] = utils.scaler(train_x[:,[2,5,8]], 'kdp')
    vali_x[:, [2,5,8]] = utils.scaler(vali_x[:, [2,5,8]], 'kdp')

    train_y = train_y[:, 0].reshape(-1,1)
    vali_y = vali_y[:, 0].reshape(-1,1)
    train_y[:,0] = utils.scaler(train_y[:,0], 'rr')
    vali_y[:,0] = utils.scaler(vali_y[:,0], 'rr')



    '''数据加载'''
    train = utils.loader(train_x, train_y, device, 64)
    vali = utils.loader(vali_x, vali_y, device, 64)

    return train, vali


def process_for_train_2399_1(device):
    '''加载:(ele1, ele2, ele3)'''
    dataset_train = np.load('../dataset-3-9.npz')
    train_x = dataset_train['x_train'].astype(np.float32)
    train_y = dataset_train['y_train'].astype(np.float32)
    vali_x = dataset_train['x_vali'].astype(np.float32)
    vali_y = dataset_train['y_vali'].astype(np.float32)



    '''裁剪数据和归一化'''
    train_x[:,[0,3,6]] = utils.scaler(train_x[:,[0,3,6]], 'ref')
    vali_x[:, [0,3,6]] = utils.scaler(vali_x[:, [0,3,6]], 'ref')
    train_x[:,[2,5,8]] = utils.scaler(train_x[:,[2,5,8]], 'kdp')
    vali_x[:, [2,5,8]] = utils.scaler(vali_x[:, [2,5,8]], 'kdp')

    train_y = train_y[:, 0].reshape(-1,1)
    vali_y = vali_y[:, 0].reshape(-1,1)
    train_y[:,0] = utils.scaler(train_y[:,0], 'rr')
    vali_y[:,0] = utils.scaler(vali_y[:,0], 'rr')

    '''数据重排'''
    train_x1 = np.zeros((len(train_x), 2,3,9,9))
    train_x1[:,0] = train_x[:, [0,3,6]]
    train_x1[:,1] = train_x[:, [2,5,8]]
    vali_x1 = np.zeros((len(vali_x), 2,3,9,9))
    vali_x1[:,0] = vali_x[:, [0,3,6]]
    vali_x1[:,1] = vali_x[:, [2,5,8]]

    '''数据加载'''
    train = utils.loader(train_x1, train_y, device, 64)
    vali = utils.loader(vali_x1, vali_y, device, 64)

    return train, vali

def process_for_train_3399_1(device):
    '''加载:(ele1, ele2, ele3)'''
    dataset_train = np.load('../dataset-3-9.npz')
    train_x = dataset_train['x_train'].astype(np.float32)
    train_y = dataset_train['y_train'].astype(np.float32)
    vali_x = dataset_train['x_vali'].astype(np.float32)
    vali_y = dataset_train['y_vali'].astype(np.float32)



    '''裁剪数据和归一化'''
    train_x[:,[0,3,6]] = utils.scaler(train_x[:,[0,3,6]], 'ref')
    vali_x[:, [0,3,6]] = utils.scaler(vali_x[:, [0,3,6]], 'ref')
    train_x[:,[1,4,7]] = utils.scaler(train_x[:,[1,4,7]], 'zdr')
    vali_x[:, [1,4,7]] = utils.scaler(vali_x[:, [1,4,7]], 'zdr')
    train_x[:,[2,5,8]] = utils.scaler(train_x[:,[2,5,8]], 'kdp')
    vali_x[:, [2,5,8]] = utils.scaler(vali_x[:, [2,5,8]], 'kdp')

    train_y = train_y[:, 0].reshape(-1,1)
    vali_y = vali_y[:, 0].reshape(-1,1)
    train_y[:,0] = utils.scaler(train_y[:,0], 'rr')
    vali_y[:,0] = utils.scaler(vali_y[:,0], 'rr')

    '''数据重排'''
    train_x1 = np.zeros((len(train_x), 3,3,9,9))
    train_x1[:,0] = train_x[:, [0,3,6]]
    train_x1[:,1] = train_x[:, [1,4,7]]
    train_x1[:,2] = train_x[:, [2,5,8]]
    vali_x1 = np.zeros((len(vali_x), 3,3,9,9))
    vali_x1[:,0] = vali_x[:, [0,3,6]]
    vali_x1[:,1] = vali_x[:, [1,4,7]]
    vali_x1[:,2] = vali_x[:, [2,5,8]]

    '''数据加载'''
    train = utils.loader(train_x1, train_y, device, 64)
    vali = utils.loader(vali_x1, vali_y, device, 64)

    return train, vali



if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True # 加速：训练测试都行
    # 配置日志记录器
    logging.basicConfig(
        filename='./train-CNNQPE.log',                  # 日志文件名
        level=logging.INFO,                   # 记录 INFO 及以上级别的日志
        format='%(asctime)s---%(message)s',   # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S'           # 时间格式
    )
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("使用的设备:", device)

    # 配置路径
    path_save = './model/based_on_202407/{}'.format('CNNQPE-999-1-vlr-wmse-new_scaler-new_stop_3')
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    

    '''模型和加载器'''
    # model = ResQPE().to(device)
    # train, vali = process_for_train_3399_1(device)
    model = CNNQPE(9,1).to(device)
    # train, vali = process_for_train_999_1(device)
    # logging.info('Model and Data loaded')
    # '''损失函数'''
    # # loss_func = torch.nn.MSELoss()
    # loss_func = WeightedMSELoss()
    # '''优化器'''
    # optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4, weight_decay = 1e-4)
    # '''变化学习率'''
    # from torch.optim.lr_scheduler import StepLR
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    # '''训练'''
    # utils.training_process(train, vali, model, loss_func, optimizer, path_save, scheduler)    
    # logging.info("Done!")


    '''快速测试'''
    from test import fast_test
    model.load_state_dict(torch.load(path_save + '/' + "model.pth"))#,map_location=torch.device('cpu')))
    fast_test(path_save, model, device, 'cnn')