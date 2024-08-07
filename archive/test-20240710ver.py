import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import utils
import my.mytools as mt

import datetime

path_save = './model/based_on_202407/{}'.format('240710-cnn-3prv-05per20-maeloss-ver2')

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备:", device)

# input(f'res will be stored in:\n{path_save}\nshall we go on[y/n]?')


def qpe(data, center):
    ref = data[:,0, center, center]
    zdr = data[:,1, center, center]
    kdp = data[:,2, center, center]
    refup = 10**(ref*0.1)
    zdrup = 10**(zdr*0.1)

    a1 = 0.0576; b1 = 0.557
    a2 = 15.421; b2 = 0.817
    a3 = 0.0059; b3 = 0.994;c3 = -4.929
    a4 = 26.778; b4 = 0.946;c4 = -1.249



    rr1 = a1*refup**b1
    rr2 = a2*kdp**b2
    rr3 = a3*refup**b3*zdrup**c3
    rr4 = a4*kdp**b4*zdrup**c4

    return rr1, rr2, rr3, rr4

def apply(data):
    refup = 10**(data[:, 0]*0.1)
    meanup = refup.mean(axis=(1,2))
    mean = 10*np.log10(meanup)
    loc = mean >= 15
    test_x = data[loc]

    test_x = utils.scaler(test_x, 'ref').astype(np.float32)
    test_x = torch.from_numpy(test_x)

    model = CNN_3prv()
    model.load_state_dict(torch.load(path_save + '/' + "cnn.pth"))#,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
    
    pred = pred.view(-1).detach().numpy()
    pred = utils.scaler(pred, 'rr', 1)

    rainrate = np.zeros(len(data))
    rainrate[loc] = pred
    return rainrate

if __name__ == "__main__":
    
    count = 0
    for file in os.listdir('../'):
        if file.endswith('npz') and '2019' in file:
            print(file)
            f = np.load(f'../{file}')
            data = f['data'][:, [3,4,5], 30-4:30+4+1, 30-4:30+4+1]
            ts = [datetime.datetime.strptime(ts, '%Y%m%d%H%M') for ts in f['ts']]

            stnm = file[5:10]
            if count == 0:
                radar = pd.DataFrame(index=ts)
                radar1 = radar.copy()
                radar2 = radar.copy()
                radar3 = radar.copy()
                radar4 = radar.copy()

                radar.loc[ts, stnm] = apply(data)
                rr1, rr2, rr3, rr4 = qpe(data, 4)
                radar1.loc[ts, stnm] = rr1
                radar2.loc[ts, stnm] = rr2
                radar3.loc[ts, stnm] = rr3
                radar4.loc[ts, stnm] = rr4
                count += 1
            else:
                radar.loc[ts, stnm] = apply(data)
                rr1, rr2, rr3, rr4 = qpe(data, 4)
                radar1.loc[ts, stnm] = rr1
                radar2.loc[ts, stnm] = rr2
                radar3.loc[ts, stnm] = rr3
                radar4.loc[ts, stnm] = rr4
                count += 1
            # break
    radar.to_csv(f'{path_save}/rainrate-dl.csv')
    radar1.to_csv(f'{path_save}/rainrate-ref.csv')
    radar2.to_csv(f'{path_save}/rainrate-kdp.csv')
    radar3.to_csv(f'{path_save}/rainrate-refzdr.csv')
    radar4.to_csv(f'{path_save}/rainrate-kdpzdr.csv')

    radar = mt.readcsv(f'{path_save}/rainrate-dl.csv', isrr=3, mask=1, acc='H')
    radar1 = mt.readcsv(f'{path_save}/rainrate-ref.csv', isrr=3, mask=1, acc='H')
    radar2 = mt.readcsv(f'{path_save}/rainrate-kdp.csv', isrr=3, mask=1, acc='H')
    radar3 = mt.readcsv(f'{path_save}/rainrate-refzdr.csv', isrr=3, mask=1, acc='H')
    radar4 = mt.readcsv(f'{path_save}/rainrate-kdpzdr.csv', isrr=3, mask=1, acc='H')
    gauge = mt.readcsv('../gauge_all.csv')

    _, _, idx, col = mt.match_df(gauge, radar)
    radar = radar.loc[idx, col]
    radar1 = radar1.loc[idx, col]
    radar2 = radar2.loc[idx, col]
    radar3 = radar3.loc[idx, col]
    radar4 = radar4.loc[idx, col]
    gauge = gauge.loc[idx, col]

    zzz = gauge.values
    aaa = radar.values
    bbb = radar1.values
    ccc = radar2.values
    ddd = radar3.values
    eee = radar4.values
    loc = (zzz>=0.1) & (aaa>=0.1) & (bbb>=0.1) & (ccc>=0.1) & (ddd>=0.1) & (eee>=0.1)
    mt.Scatter(zzz[loc], aaa[loc]).plot3(bins=[np.arange(100)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,100]]*2, show_metrics=1, draw_line=1,
                                          fpath=f'{path_save}/eval-dl.png', title='dl')
    mt.Scatter(zzz[loc], bbb[loc]).plot3(bins=[np.arange(100)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,100]]*2, show_metrics=1, draw_line=1,
                                        fpath=f'{path_save}/eval-ref.png', title='ref')
    mt.Scatter(zzz[loc], ccc[loc]).plot3(bins=[np.arange(100)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,100]]*2, show_metrics=1, draw_line=1,
                                        fpath=f'{path_save}/eval-kdp.png', title='kdp')
    mt.Scatter(zzz[loc], ddd[loc]).plot3(bins=[np.arange(100)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,100]]*2, show_metrics=1, draw_line=1,
                                        fpath=f'{path_save}/eval-refzdr.png', title='refzdr')
    mt.Scatter(zzz[loc], eee[loc]).plot3(bins=[np.arange(100)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,100]]*2, show_metrics=1, draw_line=1,
                                        fpath=f'{path_save}/eval-kdpzdr.png', title='kdpzdr')