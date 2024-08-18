import my.utils.mytools as mt
import my.utils.utils_ml as utils
import pandas as pd
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors




def mytable(zzz, aaa, bbb, ccc, ddd, eee,
          path_save, thr):
    metrics_hit = pd.DataFrame(columns=['H', 'M', 'F', 'POD', 'FAR', 'CSI'])
    metrics_hit.loc['dl'] = mt.get_metrics_hit(zzz, aaa, thr)
    metrics_hit.loc['ref'] = mt.get_metrics_hit(zzz, bbb, thr)
    metrics_hit.loc['kdp'] = mt.get_metrics_hit(zzz, ccc, thr)
    metrics_hit.loc['refzdr'] = mt.get_metrics_hit(zzz, ddd, thr)
    metrics_hit.loc['kdpzdr'] = mt.get_metrics_hit(zzz, eee, thr)
    metrics_hit.to_csv(f'{path_save}/hit-{thr}.csv')

def process_for_train_393(device):
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


