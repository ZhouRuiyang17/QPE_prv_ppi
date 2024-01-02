import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime

xxx = np.load(r"E:\QPE_prv_ppi_2_99\dataset20240101\20240101\test_x.npy")
yyy = np.load(r"E:\QPE_prv_ppi_2_99\dataset20240101\20240101\test_y.npy")

plt.ion()
plt.show()

num = 12
for i in range(6):
    x = xxx[num, i]
    y = yyy[num]
    if i==0 or i==1:
        aaa = mt.colorbar(x, 'ref')
    elif i==2 or i==3:
        aaa = mt.colorbar(x, 'zdr') 
    elif i==4 or i==5:
        aaa = mt.colorbar(x, 'kdp')
    plt.pcolormesh(x, cmap = aaa[0], norm = aaa[1])
    plt.colorbar()
    plt.show()
    print(x[4,4], y)


xup = 10**(xxx[:,1,4,4]/10)
y = yyy
a,b = 0.03468, 0.5869
plt.scatter(a*xup**b, y)

a,b = 14.93, 0.83
x = xxx[:,5,4,4]
plt.scatter(a*x**b, y)