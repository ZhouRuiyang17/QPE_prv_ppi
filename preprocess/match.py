import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime


# ----雷达
dataset_raw = np.load(r"D:\data\dataset\prv_ppi\raw\dataset.npy")[:, [1,3,5]]
dataset_more = np.load(r"D:\data\dataset\prv_ppi\raw\dataset_2018rain_145_zry.npy")
plt.scatter(dataset_raw[:,0,4,4], dataset_raw[:,2,4,4])
plt.scatter(dataset_more[:,0,4,4], dataset_more[:,2,4,4])
plt.ylim(0,10)
plt.xlim(0,80)
#%%
dataset_raw = np.concatenate((dataset_raw, dataset_more))
'''去除nan值'''
loc = np.isnan(dataset_raw)
dataset_raw[loc] = -999
'''筛选反射率>15dbz的地方'''
'''2024.5.9 only use 1.45 data'''
ref1 = dataset_raw[:, 0].reshape(-1, 81)
# ref2 = dataset_raw[:, 1].reshape(-1, 81)
mean1 = 10*np.log10(np.mean(10**(ref1/10), axis = 1))
# mean2 = 10*np.log10(np.mean(10**(ref2/10), axis = 1))
loc = (mean1 > 15)# | (mean2 > 15)
dataset = dataset_raw[loc]
'''为匹配数据做准备'''
index_raw = pd.read_csv(r"D:\data\dataset\prv_ppi\raw\index.csv", index_col = 0)
index_more = pd.read_csv(r"D:\data\dataset\prv_ppi\raw\index_2018rain_145_zry.csv", index_col = 0)
index_raw = pd.concat([index_raw, index_more], axis=0)

index = index_raw.loc[loc]
idx = pd.to_datetime(index['0']).to_list()
clm = index['1'].to_list()

#%%
# ----降雨强度
rainrate_raw = pd.read_csv(r"D:\data\beijing\dsd\RR_all.csv",index_col = 0)
'''数据匹配'''
rainrate_raw.index = pd.to_datetime(rainrate_raw.index)
rainrate = []
for i in range(len(clm)):
    ts = idx[i]
    st = str(clm[i])
    try:
        rainrate += [rainrate_raw.loc[ts, st]]
    except:
        rainrate += [0]    
rainrate = np.array(rainrate)
'''去除nan值'''
loc = np.isnan(rainrate)
rainrate[loc] = 0


# =============================================================================
# '''check: 发现很烂'''
# # prv = 10**(dataset/10)
# # a,b = 0.03468, 0.5869
# prv = dataset
# a,b = 14.93, 0.83
# rainrate_radar = a*prv[:, 5, 4, 4]**b
# loc = (rainrate > 0.1) & (rainrate_radar > 0.1)
# scatter = mt.Scatter(rainrate[loc], rainrate_radar[loc])
# scatter.plot3(bins = [np.arange(0, 100)]*2, lim=[[0,100]]*2, show_metrics=1, draw_line=1)
# '''check: 雨量分布'''
# loc = np.where(rainrate == 0)
# plt.hist(rainrate, bins = [0,0.1,1,5,10,20,30,40,50,100,200])
# plt.text(10,8000, 'num of 0 is {}, {}%'.format(len(loc[0]), len(loc[0])/len(rainrate)*100))
# plt.show()
# =============================================================================

# ----合并：只要降雨强度大于0.1的
loc = rainrate > 0.1
features = dataset[loc]
labels = rainrate[loc]
dist, _, _ = plt.hist(labels, bins = np.arange(0,102,2), log=1)
plt.show()

np.save(r"D:\data\dataset\prv_ppi\dataset20240509\features.npy", features)
np.save(r"D:\data\dataset\prv_ppi\dataset20240509\labels.npy", labels)
