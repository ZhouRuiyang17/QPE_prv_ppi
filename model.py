import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime

import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
# =============================================================================
#         self.net = nn.Sequential(
#             # 卷积层
#             nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2),
#             # 拉平
#             nn.Flatten(),
#             # 全连接
#             nn.Linear(6*6*64, 512),
#             nn.ReLU(),
#             nn.Dropout(),
#             # 全连接
#             nn.Linear(512, 32),
#             nn.ReLU(),
#             nn.Dropout(),
#             # 输出
#             nn.Linear(32, 1),
#             nn.ReLU()
#             )
# =============================================================================
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            )
        self.flt = nn.Sequential(
            nn.Flatten(),
            )
        self.fc = nn.Sequential(
            nn.Linear(8*3*3, 512),
            nn.ReLU(),
            nn.Dropout(),
            )
        self.fc2 = nn.Sequential(
            # 全连接
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(),
            )
        self.out = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
            )
        

    def forward(self, x):
        # pred = self.net(x)
        
        x = self.conv(x)
        x = self.flt(x)
        x = self.fc(x)
        x = self.fc2(x)
        pred = self.out(x)
        return pred
