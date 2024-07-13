import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn

class CNN_tian_re(nn.Module):
    def __init__(self):
        super(CNN_tian_re, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.fc1 = nn.Sequential(
            nn.Linear(64*5*5, 512),
            nn.ReLU(),
            nn.Dropout(0.7)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.7)
            )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            )

    def forward(self, x):        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flt(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

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
            nn.ReLU(),
            nn.Dropout(0.7)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.7)
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
    def __init__(self, num_input_layers, num_output):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_input_layers, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # nn.BatchNorm2d(32),
            nn.ReLU(),            
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.out = nn.Sequential(
            nn.Linear(256, num_output),
            )
        

    def forward(self, x):        
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flt(x)
        x = self.out(x)
        return x

class CNN_3prv(nn.Module):
    def __init__(self):
        super(CNN_3prv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # nn.BatchNorm2d(32),
            nn.ReLU(),            
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            )
        

    def forward(self, x):        
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flt(x)
        x = self.out(x)
        return x

class CNN_9prv(nn.Module):
    def __init__(self):
        super(CNN_9prv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # nn.BatchNorm2d(32),
            nn.ReLU(),            
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            )
        

    def forward(self, x):        
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flt(x)
        x = self.out(x)
        return x

class CNN_3prv_pool(nn.Module):
    def __init__(self):
        super(CNN_3prv_pool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.AvgPool2d(2,1),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # nn.BatchNorm2d(32),
            nn.AvgPool2d(2,1),
            nn.ReLU(),            
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.AvgPool2d(2,1),
            nn.ReLU(),
            )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.out = nn.Sequential(
            nn.Linear(4*64, 1),
            )
        

    def forward(self, x):        
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = self.flt(x)
        x = self.out(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),            
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            )
        self.out = nn.Sequential(
            nn.Linear(64, 1),
            )
        

    def forward(self, x):        
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flt(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

class CNN_ver3(nn.Module):
    def __init__(self):
        super(CNN_ver3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # nn.BatchNorm2d(32),
            nn.ReLU(),            
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.fc1 = nn.Sequential(
            nn.Linear(128*3*3, 512),
            nn.ReLU(),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            )
        self.out = nn.Sequential(
            nn.Linear(128, 1),
            )
        

    def forward(self, x):        
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flt(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

class CNN_pad_ave(nn.Module):
    def __init__(self):
        super(CNN_pad_ave, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 1),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 1),
            nn.ReLU(),            
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 1),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 1),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 1),
            nn.ReLU(),
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 1),
            nn.ReLU(),
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 1),
            nn.ReLU(),
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 1),
            nn.ReLU(),
            )
        self.flt = nn.Sequential(
            nn.Flatten()
            )
        self.out = nn.Sequential(
            nn.Linear(1024, 1),
            )
        

    def forward(self, x):        
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.flt(x)
        x = self.out(x)
        return x

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



