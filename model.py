import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn

class CNN_tian(nn.Module):
    def __init__(self):
        super(CNN_tian, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.fc1 = nn.Sequential(
            nn.Linear(32*7*7, 1024),
            nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
            )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            )

    def forward(self, x):        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flt(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),            
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.out = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU()
            )
        

    def forward(self, x):        
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flt(x)
        pred = self.out(x)
        return pred

# =============================================================================
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(),
#             )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(),            
#             )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),
#             )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
#             )
#         self.flt = nn.Sequential(
#             nn.Flatten()
#             )
#         self.out = nn.Sequential(
#             nn.Linear(128, 1),
#             nn.ReLU()
#             )
#         
# 
#     def forward(self, x):        
#         x = self.conv(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.flt(x)
#         # x = self.fc(x)
#         # x = self.fc2(x)
#         pred = self.out(x)
#         return pred
# =============================================================================

# ----ver6
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
# # =============================================================================
# #         self.net = nn.Sequential(
# #             # 卷积层
# #             nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1),
# #             nn.LeakyReLU(),
# #             nn.AvgPool2d(kernel_size=2),
# #             # 拉平
# #             nn.Flatten(),
# #             # 全连接
# #             nn.Linear(6*6*64, 512),
# #             nn.LeakyReLU(),
# #             nn.Dropout(),
# #             # 全连接
# #             nn.Linear(512, 32),
# #             nn.LeakyReLU(),
# #             nn.Dropout(),
# #             # 输出
# #             nn.Linear(32, 1),
# #             nn.LeakyReLU()
# #             )
# # =============================================================================
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3),
#             nn.LeakyReLU(),
#             nn.AvgPool2d(kernel_size=2),
#             )
#         self.flt = nn.Sequential(
#             nn.Flatten(),
#             )
#         self.fc = nn.Sequential(
#             nn.Linear(64*3*3, 512),
#             nn.LeakyReLU(),
#             nn.Dropout(),
#             )
#         self.fc2 = nn.Sequential(
#             # 全连接
#             nn.Linear(512, 32),
#             nn.LeakyReLU(),
#             nn.Dropout(),
#             )
#         self.out = nn.Sequential(
#             nn.Linear(32, 1),
#             nn.ReLU()
#             )
        

#     def forward(self, x):
#         # pred = self.net(x)
        
#         x = self.conv(x)
#         x = self.flt(x)
#         x = self.fc(x)
#         x = self.fc2(x)
#         pred = self.out(x)
#         return pred
