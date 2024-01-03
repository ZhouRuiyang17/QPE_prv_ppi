import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime
import torch
import torch.nn as nn
# conv = nn.Conv2d(1,1,3)
def conv(image):
    print(image)
    
    # 定义一个卷积核（滤波器）
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.float32)*0.25
    
    # 获取输入图像和卷积核的尺寸
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # 计算卷积后的输出图像的尺寸
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # 初始化输出图像
    output_image = np.zeros((output_height, output_width), dtype=np.float32)
    
    # 执行卷积操作
    for i in range(output_height):
        for j in range(output_width):
            output_image[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    # 打印卷积结果
    print(output_image)
    return output_image

xxx = np.load(r"E:\QPE_prv_ppi_2_99\dataset20240101\20240101\test_x.npy")
yyy = np.load(r"E:\QPE_prv_ppi_2_99\dataset20240101\20240101\test_y.npy")

plt.ion()
plt.show()

num = 100
for i in range(6):
    x = xxx[num, i]
    # x = torch.Tensor(x).view(-1,9,9)
    y = yyy[num]
    if i==0 or i==1:
        aaa = mt.colorbar(x.copy(), 'ref')
    elif i==2 or i==3:
        aaa = mt.colorbar(x.copy(), 'zdr') 
    elif i==4 or i==5:
        aaa = mt.colorbar(x.copy(), 'kdp')
    plt.pcolormesh((x), cmap = aaa[0], norm = aaa[1])
    plt.colorbar()
    plt.show()
    plt.pcolormesh(conv(x), cmap = aaa[0], norm = aaa[1])
    plt.colorbar()
    plt.show()
    print(x[4,4], y)


# xup = 10**(xxx[:,1,4,4]/10)
# y = yyy
# a,b = 0.03468, 0.5869
# plt.scatter(a*xup**b, y)

# a,b = 14.93, 0.83
# x = xxx[:,5,4,4]
# plt.scatter(a*x**b, y)