import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import utils



if __name__ == "__main__":
    

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用的设备:", device)
    
    # ----数据归一并tensor化
    features = []
    features = utils.scaler(features, 'ref')
    features = torch.from_numpy(features)

    # ----加载模型并指定设备
    model = CNN()
    model.load_state_dict(torch.load('./model/20240328-9-cnn 6prv-vlr02-check stop when 100 epoch/cnn.pth', map_location=torch.device('cpu')))
    
    # ----计算                      
    model.eval()
    with torch.no_grad():
        pred = model(features)
    
    # ----结果numpy化并逆归一
    pred = pred.view(-1).detach().numpy()
    pred = utils.scaler(pred, 'rr', 1)

