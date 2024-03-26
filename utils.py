import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
'''数据'''
def scaler(data, dtype, reverse = False):
    mins = {'ref':0,
            'zdr':0,
            'kdp':0,
            'rr':0}
    maxs = {'ref':70,
            'zdr':10,
            'kdp':10,
            'rr':100}
    
    if not reverse:
        data_new = (data - mins[dtype]) / (maxs[dtype] - mins[dtype])
        data_new[data_new<0] = 0
        data_new[data_new>1] = 1
    else:
        data_new = (maxs[dtype] - mins[dtype]) * data + mins[dtype]
        
    return data_new

def spliter(x, y, ratio):
    x = x.copy()
    y = y.copy()
    test_size = ratio[-1]/sum(ratio)
    vali_size = ratio[1]/sum(ratio[:-1])
    
    from sklearn.model_selection import train_test_split
    x1     , x_test , y1     , y_test  = train_test_split(x , y,  test_size = test_size)
    x_train, x_vali,  y_train, y_vali  = train_test_split(x1, y1, test_size = vali_size)
    return [x_train, x_vali, x_test, y_train, y_vali, y_test]

'''训练'''
def loader(x, y, batch_size = 64):
    x, y = x.copy().astype(np.float32), y.copy().astype(np.float32)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = 1)
    
    return dataloader

def early_stop(loss_vali, num_check):
    x = np.arange(num_check)
    y = loss_vali[-num_check:]
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    if slope > 0:
        return 1
    else:
        return 0