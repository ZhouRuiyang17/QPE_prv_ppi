import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
import utils


# ----归一
features = np.load(r"D:\data\dataset\prv_ppi\dataset20240325\features.npy")[:, :6]
labels = np.load(r"D:\data\dataset\prv_ppi\dataset20240325\labels.npy")
plt.hist(features[:,1, 4,4], bins = np.arange(100));plt.show()
plt.hist(labels, bins = np.arange(100));plt.show()
features[:,0:2] = utils.scaler(features[:,0:2], 'ref')
features[:,2:4] = utils.scaler(features[:,2:4], 'zdr')
features[:,4:6] = utils.scaler(features[:,4:6], 'kdp')
labels = utils.scaler(labels, 'rr')
plt.hist(features[:,1, 4,4], bins = np.arange(100)/100);plt.show()
plt.hist(labels, bins = np.arange(100)/100);plt.show()


# ----分类、划分、合并
edge = list(np.arange(0, 1, 0.1)) + [1]
train_x, train_y = [], []
vali_x, vali_y = [], []
test_x, test_y = [], []
for i, _ in enumerate(edge[:-1]):
    loc = np.where((labels > edge[i]) & (labels <= edge[i+1]))[0]
    if len(loc) > 0:
        res = utils.spliter(features[loc], labels[loc], [7,1,2])
        train_x += list(res[0]); train_y += list(res[3])
        vali_x += list(res[1]); vali_y += list(res[4])
        test_x += list(res[2]); test_y += list(res[5])
    else:
        print(edge[i],edge[i+1])
train_x = np.array(train_x)
train_y = np.array(train_y)
vali_x = np.array(vali_x)
vali_y = np.array(vali_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

# ----打乱
np.random.seed(42)

index = np.arange(len(train_y))
plt.scatter(train_x[:,5,4,4], train_y);plt.show()
np.random.shuffle(index)
train_x = np.array(train_x)[index]
train_y = np.array(train_y)[index]
# np.random.shuffle(train_x)
# np.random.shuffle(train_y)
plt.scatter(train_x[:,5,4,4], train_y);plt.show()


index = np.arange(len(vali_y))
np.random.shuffle(index)
vali_x = np.array(vali_x)[index]
vali_y = np.array(vali_y)[index]

index = np.arange(len(test_y))
np.random.shuffle(index)
test_x = np.array(test_x)[index]
test_y = np.array(test_y)[index]

np.save(r'E:\QPE_prv_ppi_2_99\dataset\20240326\train_x.npy', train_x)
np.save(r'E:\QPE_prv_ppi_2_99\dataset\20240326\train_y.npy', train_y)
np.save(r'E:\QPE_prv_ppi_2_99\dataset\20240326\vali_x.npy', vali_x)
np.save(r'E:\QPE_prv_ppi_2_99\dataset\20240326\vali_y.npy', vali_y)
np.save(r'E:\QPE_prv_ppi_2_99\dataset\20240326\test_x.npy', test_x)
np.save(r'E:\QPE_prv_ppi_2_99\dataset\20240326\test_y.npy', test_y)

#%%

# ----审视
train_x = np.load(r'E:\QPE_prv_ppi_2_99\dataset\20240326\train_x.npy')
train_y = np.load(r'E:\QPE_prv_ppi_2_99\dataset\20240326\train_y.npy')
vali_x = np.load(r'E:\QPE_prv_ppi_2_99\dataset\20240326\vali_x.npy')
vali_y = np.load(r'E:\QPE_prv_ppi_2_99\dataset\20240326\vali_y.npy')
test_x = np.load(r'E:\QPE_prv_ppi_2_99\dataset\20240326\test_x.npy')
test_y = np.load(r'E:\QPE_prv_ppi_2_99\dataset\20240326\test_y.npy')

# dist, _, _ = plt.hist(train_y, bins = np.arange(0,102,10)/100)
# plt.show()
# dist, _, _ = plt.hist(train_y, bins = np.array([0,1,3,5,10,20,30,40,50,70,100])/100)
# plt.show()
dist, _, _ = plt.hist(np.log10(utils.scaler(train_y, 'rr', 1)), bins = np.arange(-2,2,0.1))
plt.show()

dist, _, _ = plt.hist(train_x[:,1,4,4], bins = np.arange(0,102,2)/100)
plt.show()
dist, _, _ = plt.hist(train_x[:,3,4,4], bins = np.arange(0,10,0.5)/10)
plt.show()
dist, _, _ = plt.hist(train_x[:,5,4,4], bins = np.arange(0,10,0.5)/10)
plt.show()

# utils.Scatter(train_x[:,1,4,4],train_x[:,5,4,4]).plot3(bins = [np.arange(0,1,0.05)]*2)
# utils.Scatter(train_x[:,1,4,4],train_y).plot3(bins = [np.arange(0,1,0.05)]*2)
# plt.scatter(train_x[:,1,4,4],np.log10(utils.scaler(train_y, 'rr', 1)))
