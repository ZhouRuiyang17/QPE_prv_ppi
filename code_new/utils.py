import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def spliter(x, y, ratio):
    x = x.copy()
    y = y.copy()
    test_size = ratio[-1] / sum(ratio)
    vali_size = ratio[1] / sum(ratio[:-1])

    from sklearn.model_selection import train_test_split
    x1, x_test, y1, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_vali, y_train, y_vali = train_test_split(x1, y1, test_size=vali_size)
    return [x_train, x_vali, x_test, y_train, y_vali, y_test]


# ----划分
features = np.load(r"./dataset/features.npy")
labels = np.load(r"./dataset/labels.npy")
edge = list(np.arange(0, 52, 2)) + [100]
train_x, train_y = [], []
vali_x, vali_y = [], []
test_x, test_y = [], []
# weights = []
for i, _ in enumerate(edge[:-1]):
    loc = np.where((labels > edge[i]) & (labels <= edge[i + 1]))[0]
    print(loc)
    if len(loc) > 0:
        aaa = spliter(features[loc], labels[loc], [7, 1, 2])
        train_x += list(aaa[0])
        train_y += list(aaa[3])  # weights += [len(aaa[3])]
        vali_x += list(aaa[1])
        vali_y += list(aaa[4])
        test_x += list(aaa[2])
        test_y += list(aaa[5])
    else:
        print(edge[i], edge[i + 1])

random_seed = 42
np.random.seed(random_seed)
train_x = np.array(train_x)
np.random.shuffle(train_x)
train_y = np.array(train_y)
np.random.shuffle(train_y)
vali_x = np.array(vali_x)
np.random.shuffle(vali_x)
vali_y = np.array(vali_y)
np.random.shuffle(vali_y)
test_x = np.array(test_x)
np.random.shuffle(test_x)
test_y = np.array(test_y)
np.random.shuffle(test_y)

np.save(r'./dataset/train_x.npy', train_x)
np.save(r'./dataset/train_y.npy', train_y)
np.save(r'./dataset/vali_x.npy', vali_x)
np.save(r'./dataset/vali_y.npy', vali_y)
np.save(r'./dataset/test_x.npy', test_x)
np.save(r'./dataset/test_y.npy', test_y)

dist, _, _ = plt.hist(train_y, bins=np.arange(0, 102, 2))
plt.show()
dist, _, _ = plt.hist(vali_y, bins=np.arange(0, 102, 2))
plt.show()
dist, _, _ = plt.hist(test_y, bins=np.arange(0, 102, 2))
plt.show()
