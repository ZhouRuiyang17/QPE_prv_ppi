import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import data_loader
import numpy as np
import torch
from Src import models
from torch.utils import data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if __name__ == '__main__':
    device = 'cpu'

    weights_path = './weights'
    model = models.CNN(1, False).to(device)
    # key = model.load_state_dict(torch.load(r"E:\QPE_prv_ppi_2_99\code_new\weights\best\weights\weights.pth",
    #                                        map_location=torch.device('cpu')))
    key = model.load_state_dict(torch.load(weights_path + "/weights.pth"))
    print(key)
    model.eval()

    x_train, x_vali, x_test, y_train, y_vali, y_test = data_loader.get_data()

    dataset_test = data_loader.Dataset(x_test, y_test)

    test_loader = data.DataLoader(dataset_test, batch_size=64,
                                  shuffle=False, num_workers=4)

    pres_all = []
    labels_all = []

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels_all.append(labels.cpu().numpy().flatten())

        with torch.no_grad():
            outputs = model(images)
        outputs = outputs.cpu().numpy()
        pres_all.append(outputs.flatten())
    pres_all = np.concatenate(pres_all, axis=0).flatten()
    labels_all = np.concatenate(labels_all, axis=0).flatten()
    plt.hist2d(labels_all, pres_all, bins=[np.arange(0,1,0.01)]*2, norm=colors.LogNorm())
    plt.plot([0, 1], [0, 1])
    plt.title('test')
    plt.pause(0.5)
    plt.show()

    print('mae is : ', mean_absolute_error(labels_all, pres_all))
    print('mse is : ', mean_squared_error(labels_all, pres_all))
    print('r2 is : ', r2_score(labels_all, pres_all))
