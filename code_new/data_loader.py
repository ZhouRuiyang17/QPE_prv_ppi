import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from tools import ml

ratio = [7, 1, 2]
maxi = [70, 7, 7, 1, 100]
mini = [0, 0, 0, 0, 0]


def nor_label(labels):
    labels = labels - 0.1
    labels = labels / 200
    return labels


def nor(data):
    data -= np.min(data)
    data = data / np.max(data)
    return data


def scaler(datas):
    for i, data in enumerate(datas):
        datas[i] = ml.min_max(data, mini[i], maxi[i])
    return datas


def get_data():
    path_in = './dataset'
    feature = np.load(path_in + '/features.npy')[:, :6]
    label = np.load(path_in + '/labels.npy')
    label = nor_label(label)
    test_size = ratio[-1] / sum(ratio)
    vali_size = ratio[1] / sum(ratio[:-1])
    x1, x_test, y1, y_test = train_test_split(feature, label, test_size=test_size, random_state=666)
    x_train, x_vali, y_train, y_vali = train_test_split(x1, y1, test_size=vali_size, random_state=666)

    return [x_train, x_vali, x_test, y_train, y_vali, y_test]


def get_data_v1():
    path_in = './dataset'
    train_x = np.load(path_in + '/train_x.npy')
    vali_x = np.load(path_in + '/vali_x.npy')
    test_x = np.load(path_in + '/test_x.npy')

    y_train = np.load(path_in + '/train_y.npy')
    y_vali = np.load(path_in + '/vali_y.npy')
    y_test = np.load(path_in + '/test_y.npy')

    _ls = scaler([train_x[:, 0:2], train_x[:, 2:4], train_x[:, 4:6], train_x[:, 6:8], y_train])
    train_x[:, 0:2], train_x[:, 2:4], train_x[:, 4:6], train_x[:, 6:8], train_y = _ls

    _ls = scaler([vali_x[:, 0:2], vali_x[:, 2:4], vali_x[:, 4:6], vali_x[:, 6:8], y_vali])
    vali_x[:, 0:2], vali_x[:, 2:4], vali_x[:, 4:6], vali_x[:, 6:8], vali_y = _ls

    _ls = scaler([test_x[:, 0:2], test_x[:, 2:4], test_x[:, 4:6], test_x[:, 6:8]])
    test_x[:, 0:2], test_x[:, 2:4], test_x[:, 4:6], test_x[:, 6:8] = _ls
    return [train_x[:, :6], vali_x[:, :6], test_x[:, :6], y_train, y_vali, y_test]


class Dataset(data.Dataset):
    def __init__(self, images, labels):
        super().__init__()

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = self.images[index]
        label = np.array(self.labels[index])[..., None]
        data = torch.from_numpy(data).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)
        return data, label


if __name__ == '__main__':
    x_train, x_vali, x_test, y_train, y_vali, y_test = get_data_v1()
    print(x_train.shape, y_train.shape)
    print(y_train.min(), y_train.max())
    print(y_vali.min(), y_vali.max())
    print(y_test.min(), y_test.max())
