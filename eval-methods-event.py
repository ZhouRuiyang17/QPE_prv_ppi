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


aaa = mt.readcsv(f"{path}/example-res.csv", 0, isrr=3, mask=1, acc='H')
bbb = mt.readcsv(f"{path}/example-cnn.csv", 0, isrr=3, mask=1, acc='H')
ccc = mt.readcsv(f"{path}/example-ref.csv", 0, isrr=3, mask=1, acc='H')
ddd = mt.readcsv(f"{path}/example-kdp.csv", 0, isrr=3, mask=1, acc='H')
eee = mt.readcsv(f"{path}/example-refzdr.csv", 0, isrr=3, mask=1, acc='H')
fff = mt.readcsv(f"{path}/example-kdpzdr.csv", 0, isrr=3, mask=1, acc='H')
zzz = mt.readcsv(f"/home/zry/code/gauge_all.csv", 0, isrr=0, mask=1)

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



'''timeseries and distribution'''
def plot_distribution(zzz,aaa,bbb,ccc,ddd,eee,fff,savepath):

    siteinfo = pd.read_csv('../gauge_info.csv', index_col=0)
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    # plt.subplots_adjust(hspace=0.5)
    # plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(left=0, right=1.2, top=0.8, bottom=0)
    cmap, norm, cb_label, cb_tick = mt.colorbar(0,'bias')
    sc = axs[0][0].scatter(siteinfo.loc[columns, 'lon'],siteinfo.loc[columns, 'lat'], s=100, c=(aaa-zzz).mean(axis=0), cmap=cmap, norm=norm)
    axs[0][1].scatter(siteinfo.loc[columns, 'lon'],siteinfo.loc[columns, 'lat'], s=100, c=(bbb-zzz).mean(axis=0), cmap=cmap, norm=norm)
    axs[1][0].scatter(siteinfo.loc[columns, 'lon'],siteinfo.loc[columns, 'lat'], s=100, c=(ccc-zzz).mean(axis=0), cmap=cmap, norm=norm)
    axs[1][1].scatter(siteinfo.loc[columns, 'lon'],siteinfo.loc[columns, 'lat'], s=100, c=(ddd-zzz).mean(axis=0), cmap=cmap, norm=norm)
    axs[2][0].scatter(siteinfo.loc[columns, 'lon'],siteinfo.loc[columns, 'lat'], s=100, c=(eee-zzz).mean(axis=0), cmap=cmap, norm=norm)
    axs[2][1].scatter(siteinfo.loc[columns, 'lon'],siteinfo.loc[columns, 'lat'], s=100, c=(fff-zzz).mean(axis=0), cmap=cmap, norm=norm)
    for i in range(3):
        for j in range(2):
            axs[i][j].set_aspect(1)
    cbar = fig.colorbar(sc, ax=axs, location='right',shrink=0.75)
    cbar.set_ticks(cb_tick)
    cbar.set_label(cb_label) 
    fig.savefig(savepath,bbox_inches='tight')
  

def plot_timeseries(zzz,aaa,bbb,ccc,ddd,eee,fff,
                    date1, date2, sitename=None):
    '''采样'''
    zzz = zzz.loc[date1:date2]
    aaa = aaa.loc[date1:date2]
    bbb = bbb.loc[date1:date2]
    ccc = ccc.loc[date1:date2]
    ddd = ddd.loc[date1:date2]
    eee = eee.loc[date1:date2]
    fff = fff.loc[date1:date2]

    '''mask'''
    loc = (zzz>=0.1) & (aaa>=0.1) & (bbb>=0.1) & (ccc>=0.1) & (ddd>=0.1) & (eee>=0.1) & (fff>=0.1)
    zzz = zzz.where(loc, np.nan)
    aaa = aaa.where(loc, np.nan)
    bbb = bbb.where(loc, np.nan)
    ccc = ccc.where(loc, np.nan)
    ddd = ddd.where(loc, np.nan)
    eee = eee.where(loc, np.nan)
    fff = fff.where(loc, np.nan)
    plot_distribution(zzz,aaa,bbb,ccc,ddd,eee,fff, savepath=f'{path}/{date1}/{date1}-{date2}-avgbias.png')

    '''判断'''
    if sitename == 'avg':
        zzz = zzz.mean(axis=1)
        aaa = aaa.mean(axis=1)
        bbb = bbb.mean(axis=1)
        ccc = ccc.mean(axis=1)
        ddd = ddd.mean(axis=1)
        eee = eee.mean(axis=1)
        fff = fff.mean(axis=1)
    else:
        zzz = zzz[sitename]
        aaa = aaa[sitename]
        bbb = bbb[sitename]
        ccc = ccc[sitename]
        ddd = ddd[sitename]
        eee = eee[sitename]
        fff = fff[sitename]


    x = np.arange(len(zzz))
    plt.figure()
    plt.bar(x, zzz, label='Gauge',facecolor='none', edgecolor='black')
    plt.plot(x, aaa, label=titles[0],linestyle='--',marker='o')
    plt.plot(x, bbb, label=titles[1],linestyle='--',marker='o')
    plt.plot(x, ccc, label=titles[2],marker='o')
    plt.plot(x, ddd, label=titles[3],marker='o')
    plt.plot(x, eee, label=titles[4],marker='o') 
    plt.plot(x, fff, label=titles[5],marker='o')
    plt.xticks(x[::3], [f'{hour%24:00}' for hour in x][::3])

    plt.legend()
    plt.grid()
    plt.title(f'{date1}-{date2}-{sitename}')
    plt.savefig(f'{path}/{date1}/{date1}-{date2}-{sitename}.png')

plot_timeseries(zzz,aaa,bbb,ccc,ddd,eee,fff,'20190804', '20190804', 'avg')

