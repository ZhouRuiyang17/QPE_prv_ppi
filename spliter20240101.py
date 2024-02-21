import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime

# ----雷达
# =============================================================================
# dataset_raw = np.load(r"E:\QPE_prv_ppi_2_99\dataset聚合\dataset.npy")
# '''去除nan值'''
# loc = np.isnan(dataset_raw)
# dataset_raw[loc] = -999
# '''筛选中心区域反射率>15dbz的地方'''
# ref1 = dataset_raw[:, 0, 3:6, 3:6].reshape(-1, 9)
# ref2 = dataset_raw[:, 1, 3:6, 3:6].reshape(-1, 9)
# mean1 = 10*np.log10(np.mean(10**(ref1/10), axis = 1))
# mean2 = 10*np.log10(np.mean(10**(ref2/10), axis = 1))
# loc = (mean1 > 15) | (mean2 > 15)
# dataset = dataset_raw[loc]
# '''为匹配数据做准备'''
# index_raw = pd.read_csv(r"E:\QPE_prv_ppi_2_99\dataset聚合\index.csv", index_col = 0)
# index = index_raw.loc[loc]
# idx = pd.to_datetime(index['0']).to_list()
# clm = index['1'].to_list()
# 
# # ----降雨强度
# rainrate_raw = pd.read_csv(r"D:\data\RR_all.csv",index_col = 0)
# '''数据匹配'''
# rainrate_raw.index = pd.to_datetime(rainrate_raw.index)
# rainrate = []
# for i in range(len(clm)):
#     ts = idx[i]
#     st = str(clm[i])
#     try:
#         rainrate += [rainrate_raw.loc[ts, st]]
#     except:
#         rainrate += [0]    
# rainrate = np.array(rainrate)
# '''去除nan值'''
# loc = np.isnan(rainrate)
# rainrate[loc] = 0
# '''验证：发现很烂'''
# # =============================================================================
# # prv = 10**(dataset/10)
# # # a,b = 0.03468, 0.5869
# # prv = dataset
# # a,b = 14.93, 0.83
# # rainrate222 = a*prv[:, 5, 4, 4]**b
# # scatter = mt.Scatter(rainrate, rainrate222)
# # scatter.plot3(bins = [np.arange(0, 50, 0.5)]*2, lim=[[0,50]]*2, show_metrics=1, draw_line=1)
# # =============================================================================
# '''看看雨量分布'''
# loc = np.where(rainrate == 0)
# plt.hist(rainrate, bins = [0,0.1,1,5,10,20,30,40,50,100,200])
# plt.text(10,8000, 'num of 0 is {}, {}%'.format(len(loc[0]), len(loc[0])/len(rainrate)*100))
# plt.show()
# 
# # ----合并：只要降雨强度大于0.1的
# loc = rainrate > 0.1
# features = dataset[loc]
# labels = rainrate[loc]
# dist, _, _ = plt.hist(labels, bins = np.arange(0,102,2))
# plt.show()
# 
# np.save(r"D:\dataset\prv_ppi\dataset20240101\features.npy", features)
# np.save(r"D:\dataset\prv_ppi\dataset20240101\labels.npy", labels)
# =============================================================================

# ----划分
features = np.load(r"D:\data\dataset\prv_ppi\dataset20240101\features.npy")
labels = np.load(r"D:\data\dataset\prv_ppi\dataset20240101\labels.npy")
edge = list(np.arange(0, 52, 2)) + [100]
train_x, train_y = [], []
vali_x, vali_y = [], []
test_x, test_y = [], []
# weights = []
for i, _ in enumerate(edge[:-1]):
    loc = np.where((labels > edge[i]) & (labels <= edge[i+1]))[0]
    if len(loc) > 0:
        aaa = mt.spliter(features[loc], labels[loc], [7,1,2])
        train_x += list(aaa[0]); train_y += list(aaa[3]);# weights += [len(aaa[3])]
        vali_x += list(aaa[1]); vali_y += list(aaa[4])
        test_x += list(aaa[2]); test_y += list(aaa[5])
    else:
        print(edge[i],edge[i+1])

random_seed = 42
np.random.seed(random_seed)
train_x = np.array(train_x);np.random.shuffle(train_x)
train_y = np.array(train_y);np.random.shuffle(train_y)
vali_x = np.array(vali_x);np.random.shuffle(vali_x)
vali_y = np.array(vali_y);np.random.shuffle(vali_y)
test_x = np.array(test_x);np.random.shuffle(test_x)
test_y = np.array(test_y);np.random.shuffle(test_y)
# weights = np.array(weights/np.sum(weights))[::-1]

np.save(r'D:\data\dataset\prv_ppi\dataset20240101\20240221\train_x.npy', train_x)
np.save(r'D:\data\dataset\prv_ppi\dataset20240101\20240221\train_y.npy', train_y)
np.save(r'D:\data\dataset\prv_ppi\dataset20240101\20240221\vali_x.npy', vali_x)
np.save(r'D:\data\dataset\prv_ppi\dataset20240101\20240221\vali_y.npy', vali_y)
np.save(r'D:\data\dataset\prv_ppi\dataset20240101\20240221\test_x.npy', test_x)
np.save(r'D:\data\dataset\prv_ppi\dataset20240101\20240221\test_y.npy', test_y)
# np.save(r'D:\dataset\prv_ppi\dataset20240101\20240221\weights.npy', weights)
# np.save(r'D:\dataset\prv_ppi\dataset20240101\20240221\edge.npy', np.array(edge))

dist, _, _ = plt.hist(train_y, bins = np.arange(0,102,2))
plt.show()
dist, _, _ = plt.hist(vali_y, bins = np.arange(0,102,2))
plt.show()
dist, _, _ = plt.hist(test_y, bins = np.arange(0,102,2))
plt.show()


