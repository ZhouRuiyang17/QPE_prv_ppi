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


# aaa = mt.readcsv(f"{path}/example-res.csv", 0, isrr=0, mask=1, acc='no')
# bbb = mt.readcsv(f"{path}/example-cnn.csv", 0, isrr=0, mask=1, acc='no')
# ccc = mt.readcsv(f"{path}/example-ref.csv", 0, isrr=0, mask=1, acc='no')
# ddd = mt.readcsv(f"{path}/example-kdp.csv", 0, isrr=0, mask=1, acc='no')
# eee = mt.readcsv(f"{path}/example-refzdr.csv", 0, isrr=0, mask=1, acc='no')
# fff = mt.readcsv(f"{path}/example-kdpzdr.csv", 0, isrr=0, mask=1, acc='no')

# titles = ['Res Model', 'CNN Model', 'R(Z)', 'R(kdp)', 'R(Z,zdr)', 'R(kdp,zdr)']

# '''match'''
# index, columns = mt.match_df(aaa, [bbb,ccc,ddd,eee,fff])
# aaa = aaa.loc[index, columns]
# bbb = bbb.loc[index, columns]
# ccc = ccc.loc[index, columns]
# ddd = ddd.loc[index, columns]
# eee = eee.loc[index, columns]
# fff = fff.loc[index, columns]



# '''timeseries'''
# def plot_timeseries(aaa,bbb,ccc,ddd,eee,fff,
#                     date1, date2, sitename):
#     aaa1 = aaa.loc[date1:date2]
#     bbb1 = bbb.loc[date1:date2]
#     ccc1 = ccc.loc[date1:date2]
#     ddd1 = ddd.loc[date1:date2]
#     eee1 = eee.loc[date1:date2]
#     fff1 = fff.loc[date1:date2]

#     loc = (aaa1>=0.1) & (bbb1>=0.1) & (ccc1>=0.1) & (ddd1>=0.1) & (eee1>=0.1) & (fff1>=0.1)
#     aaa1 = aaa1.where(loc, np.nan)
#     bbb1 = bbb1.where(loc, np.nan)
#     ccc1 = ccc1.where(loc, np.nan)
#     ddd1 = ddd1.where(loc, np.nan)
#     eee1 = eee1.where(loc, np.nan)
#     fff1 = fff1.where(loc, np.nan)
#     # print(zzz1.max(axis=0))
#     # return 0

#     aaa1 = aaa1[sitename]
#     bbb1 = bbb1[sitename]
#     ccc1 = ccc1[sitename]
#     ddd1 = ddd1[sitename]
#     eee1 = eee1[sitename]
#     fff1 = fff1[sitename]


#     x = np.arange(len(aaa1.index))
#     plt.plot(x, aaa1, label=titles[0],linestyle='--',marker='o')
#     plt.plot(x, bbb1, label=titles[1],linestyle='--',marker='o')
#     plt.plot(x, ccc1, label=titles[2],marker='o')
#     plt.plot(x, ddd1, label=titles[3],marker='o')
#     plt.plot(x, eee1, label=titles[4],marker='o')
#     plt.plot(x, fff1, label=titles[5],marker='o')
#     xticks = [time.strftime('%H:%M') for time in aaa1.index]
#     plt.xticks(x[::5], xticks[::5])

#     plt.legend()
#     plt.grid()
#     plt.savefig(f'{path}/{date1}-{date2}-{sitename}-rr.png')
# plot_timeseries(aaa,bbb,ccc,ddd,eee,fff,'20190804 17:00', '20190804 20:00', 'A1633')

'''ppi plot'''
################### Xband
key = '20190804.18'
ls = []
for root, dirs, files in os.walk(r'/data/zry/radar/Xradar_npz_qc/BJXSY'):
    for file in files:
        if key in file:# and '20180716' not in file:
            ls += [os.path.join(root, file)]
ls = sorted(ls)
# print(ls)
i=0
for fp in ls:
    data = np.load(fp)
    print(data.files)
    data = data['data']
    mt.RADAR(data[0], 'ref', *mt.BJXSY).ppi_lonlat(1, scatters=[116.6511078,40.39972305,70], 
                                                   title = f'{key}-{i}',
                                                    lim = [116,117,40,40.8],
                                                   savepath=f'{path}/20190804/{key}-{i}.png')
    i+=3
    f=1
    # break
################### Sband
# key = '2019080418'
# ls = []
# for root, dirs, files in os.walk(r'/data/zry/radar/Sradar_npz/20190804'):
#     for file in files:
#         if key in file:# and '20180716' not in file:
#             ls += [os.path.join(root, file)]
# ls = sorted(ls)
# # print(ls)
# i=0
# for fp in ls:
#     data = np.load(fp)
#     print(data.files)
#     data = data['arr_0']
#     mt.RADAR(data, 'ref', *mt.BJ_RADAR).ppi_lonlat(1, scatters=[116.6511078,40.39972305,70], 
#                                                    title = f'{key}-{i}',
#                                                    lim = [116,117,40,40.8],
#                                                    savepath=f'{path}/20190804/{key}-{i}-S.png')
#     i+=6
#     f=1
#     # break
