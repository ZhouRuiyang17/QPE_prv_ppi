import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.utils.mytools as mt
import datetime
import matplotlib.colors as colors
# plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.size'] = 12

path = '/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse'

aaa = mt.readcsv(f"{path}/example-dl.csv", 0, isrr=3, mask=1, acc='H')
bbb = mt.readcsv(f"{path}/example-ref.csv", 0, isrr=3, mask=1, acc='H')
ccc = mt.readcsv(f"{path}/example-kdp.csv", 0, isrr=3, mask=1, acc='H')
ddd = mt.readcsv(f"{path}/example-refzdr.csv", 0, isrr=3, mask=1, acc='H')
eee = mt.readcsv(f"{path}/example-kdpzdr.csv", 0, isrr=3, mask=1, acc='H')
zzz = mt.readcsv(f"/home/zry/code/gauge_all.csv", 0, isrr=0, mask=1)

titles = ['CNN Model', 'R(Z)', 'R(kdp)', 'R(Z,zdr)', 'R(kdp,zdr)']

'''match'''
_, _, index, columns = mt.match_df(aaa, zzz)
aaa = aaa.loc[index, columns]
bbb = bbb.loc[index, columns]
ccc = ccc.loc[index, columns]
ddd = ddd.loc[index, columns]
eee = eee.loc[index, columns]
zzz = zzz.loc[index, columns]

date = '20190809'
aaa = aaa.loc[date]
bbb = bbb.loc[date]
ccc = ccc.loc[date]
ddd = ddd.loc[date]
eee = eee.loc[date]
zzz = zzz.loc[date]


# # 要删除的日期列表
# dates_to_remove = ['20170521','20170522','20170523',
#                    '20170810',
#                    '20190525','20190526','20190527']
# # 将日期转换为datetime对象
# dates_to_remove = pd.to_datetime(dates_to_remove)
# # 过滤掉要删除的日期
# aaa = aaa[~aaa.index.normalize().isin(dates_to_remove)]
# bbb = bbb[~bbb.index.normalize().isin(dates_to_remove)]
# ccc = ccc[~ccc.index.normalize().isin(dates_to_remove)]
# zzz = zzz[~zzz.index.normalize().isin(dates_to_remove)]

'''scatters'''
def plot_all_scatters(aaa, bbb, ccc, ddd, eee, zzz,
                      fig_path, met_path):
    aaa = aaa.values.flatten()
    bbb = bbb.values.flatten()
    ccc = ccc.values.flatten()
    ddd = ddd.values.flatten()
    eee = eee.values.flatten()
    zzz = zzz.values.flatten()
    loc = (zzz>=0.1) & (aaa>=0.1) & (bbb>=0.1) & (ccc>=0.1) & (ddd>=0.1) & (eee>=0.1)
    # print(len(np.where(loc==True)[0]))
    
        
    # 创建一个包含3个子图的图形
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    # plt.subplots_adjust(hspace=0.5)
    # plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(left=0, right=1.2, top=0.8, bottom=0)
    
    hd1 = mt.hist2d(axs[0], zzz[loc], aaa[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd2 = mt.hist2d(axs[1], zzz[loc], bbb[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd3 = mt.hist2d(axs[2], zzz[loc], ccc[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd4 = mt.hist2d(axs[3], zzz[loc], ddd[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd5 = mt.hist2d(axs[4], zzz[loc], eee[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    
    '''cb'''
    fig.colorbar(hd1[3], ax=axs, location='right', pad=0.05)
    '''title'''
    for i in range(len(titles)):
        axs[i].set_title(titles[i])
    '''save'''
    fig.savefig(fig_path, transparent=0,dpi=fig.dpi, bbox_inches='tight')
    
    '''metrics'''
    met = mt.get_metrics(zzz[loc], aaa[loc])
    met = pd.DataFrame(met, index=[titles[0]], columns=met.keys())
    met.loc[titles[1]] = mt.get_metrics(zzz[loc], bbb[loc])
    met.loc[titles[2]] = mt.get_metrics(zzz[loc], ccc[loc])
    met.loc[titles[3]] = mt.get_metrics(zzz[loc], ddd[loc])
    met.loc[titles[4]] = mt.get_metrics(zzz[loc], eee[loc])
    met = met[['total num', 'RMB', 'MAE', 'RMSE', 'CC']]
    met.to_csv(met_path)
    print(met)

plot_all_scatters(aaa, bbb, ccc, ddd, eee, zzz, f"{path}/example-hour-{date}-all.png", f"{path}/example-hour-{date}-all.csv")
# plot_all_scatters(aaa, bbb, ccc, ddd, eee, zzz, f"{path}/example-hour-all.png", f"{path}/example-hour-all.csv")


'''box and hist'''
def plot_box(aaa, bbb, ccc, ddd, eee, zzz, labels,
             fpbox, fphist):
    aaa = aaa.values.flatten()
    bbb = bbb.values.flatten()
    ccc = ccc.values.flatten()
    ddd = ddd.values.flatten()
    eee = eee.values.flatten()
    zzz = zzz.values.flatten()
    loc = (zzz>=0.1) & (aaa>=0.1) & (bbb>=0.1) & (ccc>=0.1) & (ddd>=0.1) & (eee>=0.1)

    plt.figure()
    plt.boxplot([aaa[loc]-zzz[loc], bbb[loc]-zzz[loc], ccc[loc]-zzz[loc], ddd[loc]-zzz[loc], eee[loc]-zzz[loc]],
                showfliers=0, labels=titles, showmeans=1,
                )
    plt.grid()
    plt.savefig(fpbox)
    
    plt.figure()
    plt.hist(aaa[loc]-zzz[loc], bins=np.arange(-10, 11, 1), alpha=0.5, label=labels[0])
    plt.hist(bbb[loc]-zzz[loc], bins=np.arange(-10, 11, 1), alpha=0.5, label=labels[1])
    plt.hist(ccc[loc]-zzz[loc], bins=np.arange(-10, 11, 1), alpha=0.5, label=labels[2])
    plt.hist(ddd[loc]-zzz[loc], bins=np.arange(-10, 11, 1), alpha=0.5, label=labels[3])
    plt.hist(eee[loc]-zzz[loc], bins=np.arange(-10, 11, 1), alpha=0.5, label=labels[4])
    plt.legend()
    plt.grid()
    plt.savefig(fphist)

plot_box(aaa, bbb, ccc, ddd, eee, zzz, titles, f"{path}/example-hour-{date}-box.png", f"{path}/example-hour-{date}-hist.png")
# plot_box(aaa, bbb, ccc, ddd, eee, zzz, titles, f"{path}/example-hour-box.png", f"{path}/example-hour-hist.png")


    

'''distribution'''
def summary(ls):
    count = 0
    lenth = len(ls)
    for fp in ls:
        if count == 0:
            data = np.load(fp) * 3/60
            count += 1
        else:
            data += np.load(fp) * 3/60
            count += 1

    print(f'{count} / {lenth}')
    return data

def compare_event(gauge, accumulation):
    
    sites_used = gauge.columns
    gaugesum = gauge.sum()
    loc = (gaugesum >= 10)

    '''site info'''
    siteinfo = pd.read_csv(r"../gauge_info.csv",index_col=0).sort_values(by='stnm')
    lons = siteinfo.loc[sites_used, 'lon']
    lats = siteinfo.loc[sites_used, 'lat']


    mt.RADAR(accumulation[4], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'dl', save=f'{path}/example-{date}-dl.png')
    mt.RADAR(accumulation[0], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'ref', save=f'{path}/example-{date}-ref.png')
    mt.RADAR(accumulation[1], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'kdp', save=f'{path}/example-{date}-kdp.png')
    mt.RADAR(accumulation[2], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'refzdr', save=f'{path}/example-{date}-refzdr.png')
    mt.RADAR(accumulation[3], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'kdpzdr', save=f'{path}/example-{date}-kdpzdr.png')

path_qpe = r'/data/zry/radar/Xradar_npy_qpe/BJXSY'
ls = []
for file in os.listdir(path_qpe):
    if date in file:
        ls += [path_qpe + '/' + file]
compare_event(zzz, summary(ls))