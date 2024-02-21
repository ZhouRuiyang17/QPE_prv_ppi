import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime
import torch
import torch.nn as nn

features = np.load(r"E:\QPE_prv_ppi_2_99\dataset20240101\features.npy")
labels = np.load(r"E:\QPE_prv_ppi_2_99\dataset20240101\labels.npy")

# =============================================================================
# ref = features[:,1,4,4]
# loc = (ref>0) & (labels>0)
# b = 1/1.4; a = (1/300)**b; pred = a*(10**(ref/10))**b
# aaa = mt.Scatter(ref[loc], labels[loc]); aaa.plot3(bins=[np.arange(0,70,0.7), np.arange(0,100,1)], lines = [ref[loc], pred[loc]])
# ref = 10**(ref/10); ref = np.log10(ref); labels = np.log10(labels)
# loc = (ref>0) & (labels>-1)
# aaa = mt.Scatter(ref[loc], labels[loc]); aaa.plot3(bins=[np.arange(0,7,0.07), np.arange(-1,2,0.03)])
# =============================================================================

# =============================================================================
# kdp = features[:,5,4,4]
# loc = (kdp>0.5) & (labels>0)
# b = 0.817; a = 15.421; pred = a*(kdp)**b
# aaa = mt.Scatter(kdp[loc]*10, labels[loc]); aaa.plot3(bins=[np.arange(0,100,1), np.arange(0,100,1)], lines = [kdp[loc]*10, pred[loc]],label=['10*kdp', 'rr'])
# plt.scatter(kdp[loc], labels[loc])
# plt.scatter(kdp[loc], pred[loc])
# =============================================================================

plt.hist(labels, bins = np.arange(0,102,2))