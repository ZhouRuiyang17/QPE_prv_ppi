import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import my.utils.utils_ml as utils
from common_tool import *

import logging


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True # 加速：训练测试都行
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("使用的设备:", device)
     
    # 配置日志记录器
    logging.basicConfig(
        filename='main.log',                  # 日志文件名
        level=logging.INFO,                   # 记录 INFO 及以上级别的日志
        format='%(asctime)s---%(message)s',   # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S'           # 时间格式
    )

    # 配置路径
    path_save = './model/based_on_202407/{}'.format('240727-cnn-9prv-3out-wmse-0818re')
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    

    '''加载器'''
    train, vali = process_for_train_393(device)
    logging.info('Data loaded')
    '''模型'''
    model = CNN(9,3).to(device)
    '''损失函数'''
    # loss_func = torch.nn.MSELoss()
    loss_func = WeightedMSELoss()
    '''优化器'''
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4, weight_decay = 1e-4)
    '''变化学习率'''
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    '''训练'''
    utils.training_process(train, vali, model, loss_func, optimizer, path_save, scheduler)    
    logging.info("Done!")
    
