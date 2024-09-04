import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNQPE(nn.Module):
    def __init__(self, num_input_layers, num_output):
        super(CNNQPE, self).__init__()
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

class ResidualLayer(nn.Module):
    '''keep size same'''
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = self.conv3(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x + residual


class ResQPE(nn.Module):
    def __init__(self):
        super(ResQPE, self).__init__()
        # Input1 branch
        self.res1 = ResidualLayer(3, 32)
        self.res2 = ResidualLayer(32, 32)
        self.conv3d = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
        # self.pool3d = nn.AvgPool3d(kernel_size=(2, 2, 2))
        
       
        self.conv2d_1 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2d = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, input1):
        # ----Input1: (3,3,9,9)
        x1 = self.res1(input1)
        # ----(32,3,9,9)
        x1 = self.res2(x1)
        # ----(32,3,9,9)
        x1 = F.relu(self.conv3d(x1))
        # ----(64,1,7,7)
        x1 = x1.squeeze()
        # ----(64,7,7)
        
        
        
        x = F.relu(self.conv2d_1(x1))
        # ----32 5 5
        x = F.relu(self.conv2d_2(x))
        # ----32 3 3
        x = self.pool2d(x)
        # ----32 1 1
        x = x.squeeze()
        # ----32
        x = F.relu(self.fc1(x))
        # ----64
        output = F.relu(self.fc2(x))
        # ----1
        
        return output

class ResNet_3399_1(nn.Module):
    def __init__(self):
        super(ResNet_3399_1, self).__init__()
        # Input1 branch
        self.res1 = ResidualLayer(1, 32)
        self.res2 = ResidualLayer(32, 32)
        self.conv3d = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
        # self.pool3d = nn.AvgPool3d(kernel_size=(2, 2, 2))
        
        # Input1 branch
        self.res1_zdr = ResidualLayer(1, 32)
        self.res2_zdr = ResidualLayer(32, 32)
        self.conv3d_zdr = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))

        # Input1 branch
        self.res1_kdp = ResidualLayer(1, 32)
        self.res2_kdp = ResidualLayer(32, 32)
        self.conv3d_kdp = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
       
        self.conv2d_1 = nn.Conv2d(64*3, 32, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2d = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, input1):
        ref = input1[:, 0].unsqueeze(1)
        zdr = input1[:, 1].unsqueeze(1)
        kdp = input1[:, 2].unsqueeze(1)
        # ----Input1: (1,3,9,9)
        x1 = self.res1(ref)
        # ----(32,3,9,9)
        x1 = self.res2(x1)
        # ----(32,3,9,9)
        x1 = F.relu(self.conv3d(x1))
        # ----(64,1,7,7)
        x1 = x1.squeeze()
        # ----(64,7,7)

        # ----Input1: (1,3,9,9)
        zdr = self.res1_zdr(zdr)
        # ----(32,3,9,9)
        zdr = self.res2_zdr(zdr)
        # ----(32,3,9,9)
        zdr = F.relu(self.conv3d_zdr(zdr))
        # ----(64,1,7,7)
        zdr = zdr.squeeze()
        # ----(64,7,7)

        # ----Input1: (1,3,9,9)
        kdp = self.res1_kdp(kdp)
        # ----(32,3,9,9)
        kdp = self.res2_kdp(kdp)
        # ----(32,3,9,9)
        kdp = F.relu(self.conv3d_kdp(kdp))
        # ----(64,1,7,7)
        kdp = kdp.squeeze()
        # ----(64,7,7)
        
        x1 = torch.cat((x1, zdr, kdp), dim=1)
        
        x = F.relu(self.conv2d_1(x1))
        # ----32 5 5
        x = F.relu(self.conv2d_2(x))
        # ----32 3 3
        x = self.pool2d(x)
        # ----32 1 1
        x = x.squeeze()
        # ----32
        x = F.relu(self.fc1(x))
        # ----64
        output = F.relu(self.fc2(x))
        # ----1
        
        return output
    

class ResNet_2399_1(nn.Module):
    def __init__(self):
        super(ResNet_2399_1, self).__init__()
        # ref branch
        self.res1 = ResidualLayer(1, 32)
        self.res2 = ResidualLayer(32, 32)
        self.conv3d = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
        # self.pool3d = nn.AvgPool3d(kernel_size=(2, 2, 2))
        

        # kdp branch
        self.res1_kdp = ResidualLayer(1, 32)
        self.res2_kdp = ResidualLayer(32, 32)
        self.conv3d_kdp = nn.Conv3d(32, 64, kernel_size=(3, 3, 3))
       
        self.conv2d_1 = nn.Conv2d(64*2, 32, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2d = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, input1):
        ref = input1[:, 0].unsqueeze(1)
        kdp = input1[:, 1].unsqueeze(1)
        # ----Input1: (1,3,9,9)
        x1 = self.res1(ref)
        # ----(32,3,9,9)
        x1 = self.res2(x1)
        # ----(32,3,9,9)
        x1 = F.relu(self.conv3d(x1))
        # ----(64,1,7,7)
        x1 = x1.squeeze()
        # ----(64,7,7)


        # ----Input1: (1,3,9,9)
        kdp = self.res1_kdp(kdp)
        # ----(32,3,9,9)
        kdp = self.res2_kdp(kdp)
        # ----(32,3,9,9)
        kdp = F.relu(self.conv3d_kdp(kdp))
        # ----(64,1,7,7)
        kdp = kdp.squeeze()
        # ----(64,7,7)
        
        x1 = torch.cat((x1, kdp), dim=1)
        
        x = F.relu(self.conv2d_1(x1))
        # ----32 5 5
        x = F.relu(self.conv2d_2(x))
        # ----32 3 3
        x = self.pool2d(x)
        # ----32 1 1
        x = x.squeeze()
        # ----32
        x = F.relu(self.fc1(x))
        # ----64
        output = F.relu(self.fc2(x))
        # ----1
        
        return output
    
# model = ResNet_ver3()
# input1 = torch.randn(10, 2, 3, 9, 9)  # Batch size of 10, (5, 11, 11) radar images
# output = model(input1)
# print('done')