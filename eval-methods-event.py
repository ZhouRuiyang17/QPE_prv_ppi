import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.utils.mytools as mt
import datetime
import matplotlib.colors as colors
# plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.size'] = 12

path = '/home/zry/code/QPE_prv_ppi/dataset/20240916eval_2models'


aaa = mt.readcsv(f"{path}/example-res.csv", 0, isrr=3, mask=1, acc='H', value_max=100)
bbb = mt.readcsv(f"{path}/example-cnn.csv", 0, isrr=3, mask=1, acc='H', value_max=100)
ccc = mt.readcsv(f"{path}/example-ref.csv", 0, isrr=3, mask=1, acc='H', value_max=100)
ddd = mt.readcsv(f"{path}/example-kdp.csv", 0, isrr=3, mask=1, acc='H', value_max=100)
eee = mt.readcsv(f"{path}/example-refzdr.csv", 0, isrr=3, mask=1, acc='H', value_max=100)
fff = mt.readcsv(f"{path}/example-kdpzdr.csv", 0, isrr=3, mask=1, acc='H', value_max=100)
zzz = mt.readcsv(f"/home/zry/code/gauge_all.csv", 0, isrr=0, mask=1, value_max=100)

titles = ['Res Model', 'CNN Model', 'R(Z)', 'R(kdp)', 'R(Z,zdr)', 'R(kdp,zdr)']

'''match'''
index, columns = mt.match_df(zzz, [aaa,bbb,ccc,ddd,eee,fff])
aaa = aaa.loc[index, columns]
bbb = bbb.loc[index, columns]
ccc = ccc.loc[index, columns]
ddd = ddd.loc[index, columns]
eee = eee.loc[index, columns]
fff = fff.loc[index, columns]
zzz = zzz.loc[index, columns]



'''timeseries'''
def plot_timeseries(zzz,aaa,bbb,ccc,ddd,eee,fff,
                    date1, date2, sitename):
    zzz1 = zzz.loc[date1:date2]
    aaa1 = aaa.loc[date1:date2]
    bbb1 = bbb.loc[date1:date2]
    ccc1 = ccc.loc[date1:date2]
    ddd1 = ddd.loc[date1:date2]
    eee1 = eee.loc[date1:date2]
    fff1 = fff.loc[date1:date2]

    loc = (zzz1>=0.1) & (aaa1>=0.1) & (bbb1>=0.1) & (ccc1>=0.1) & (ddd1>=0.1) & (eee1>=0.1) & (fff1>=0.1)
    zzz1 = zzz1.where(loc, np.nan)
    aaa1 = aaa1.where(loc, np.nan)
    bbb1 = bbb1.where(loc, np.nan)
    ccc1 = ccc1.where(loc, np.nan)
    ddd1 = ddd1.where(loc, np.nan)
    eee1 = eee1.where(loc, np.nan)
    fff1 = fff1.where(loc, np.nan)
    # print(zzz1.max(axis=0))
    # return 0

    zzz1 = zzz1[sitename]
    aaa1 = aaa1[sitename]
    bbb1 = bbb1[sitename]
    ccc1 = ccc1[sitename]
    ddd1 = ddd1[sitename]
    eee1 = eee1[sitename]
    fff1 = fff1[sitename]


    x = np.arange(24)
    plt.bar(x, zzz1, label='Gauge',facecolor='none', edgecolor='black')
    plt.plot(x, aaa1, label=titles[0],linestyle='--',marker='o')
    plt.plot(x, bbb1, label=titles[1],linestyle='--',marker='o')
    plt.plot(x, ccc1, label=titles[2],marker='o')
    plt.plot(x, ddd1, label=titles[3],marker='o')
    plt.plot(x, eee1, label=titles[4],marker='o')
    plt.plot(x, fff1, label=titles[5],marker='o')
    plt.xticks(x, [f'{hour:00}' for hour in x])

    plt.legend()
    plt.grid()
    plt.savefig(f'{path}/{date1}-{date2}-{sitename}.png')
plot_timeseries(zzz,aaa,bbb,ccc,ddd,eee,fff,'20190804', '20190804', 'A1633')

