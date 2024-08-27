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
# ddd = mt.readcsv(f"{path}/example-refzdr.csv", 0, isrr=3, mask=1, acc='H')
eee = mt.readcsv(f"{path}/example-kdpzdr.csv", 0, isrr=3, mask=1, acc='H')
zzz = mt.readcsv(f"/home/zry/code/gauge_all.csv", 0, isrr=0, mask=1)

titles = ['CNN Model', 'R(Z)', 'R(kdp)', 'R(kdp,zdr)']

'''match'''
_, _, index, columns = mt.match_df(aaa, zzz)
aaa = aaa.loc[index, columns]
bbb = bbb.loc[index, columns]
ccc = ccc.loc[index, columns]
# ddd = ddd.loc[index, columns]
eee = eee.loc[index, columns]
zzz = zzz.loc[index, columns]

date = ''
# date = '20190809'
# aaa = aaa.loc[date]
# bbb = bbb.loc[date]
# ccc = ccc.loc[date]
# # ddd = ddd.loc[date]
# eee = eee.loc[date]
# zzz = zzz.loc[date]
path = path + '/eval-0827re'
if not os.path.exists(path):
    os.makedirs(path)



'''scatters'''
def plot_all_scatters(aaa, bbb, ccc, eee, zzz,
                      fig_path, met_path):
    aaa = aaa.values.flatten()
    bbb = bbb.values.flatten()
    ccc = ccc.values.flatten()
    # ddd = ddd.values.flatten()
    eee = eee.values.flatten()
    zzz = zzz.values.flatten()
    loc = (zzz>=0.1) & (aaa>=0.1) & (bbb>=0.1) & (ccc>=0.1) & (eee>=0.1)
    # print(len(np.where(loc==True)[0]))
    
        
    # 创建一个包含3个子图的图形
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # plt.subplots_adjust(hspace=0.5)
    # plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(left=0, right=1.2, top=1, bottom=0)
    
    hd1 = mt.hist2d(axs[0][0], zzz[loc], aaa[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd2 = mt.hist2d(axs[0][1], zzz[loc], bbb[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd3 = mt.hist2d(axs[1][0], zzz[loc], ccc[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    # hd4 = mt.hist2d(axs[1][1], zzz[loc], ddd[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd5 = mt.hist2d(axs[1][1], zzz[loc], eee[loc], bins=[np.arange(61)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    
    '''cb'''
    fig.colorbar(hd1[3], ax=axs, location='right', pad=0.05)
    '''title'''
    count = 0
    for i in range(2):
        for j in range(2):
            axs[i][j].set_title(titles[count])
            count += 1
    '''save'''
    fig.savefig(fig_path, transparent=0, dpi=600, bbox_inches='tight')
    
    '''metrics'''
    met = {**mt.get_metrics(zzz[loc], aaa[loc]), **mt.get_metrics_hit(zzz,aaa,4)}
    met = pd.DataFrame(met, index=[titles[0]], columns=met.keys())
    met.loc[titles[1]] = {**mt.get_metrics(zzz[loc], bbb[loc]), **mt.get_metrics_hit(zzz,bbb,4)}
    met.loc[titles[2]] = {**mt.get_metrics(zzz[loc], ccc[loc]), **mt.get_metrics_hit(zzz,ccc,4)}
    # met.loc[titles[3]] = mt.get_metrics(zzz[loc], ddd[loc])
    met.loc[titles[3]] = {**mt.get_metrics(zzz[loc], eee[loc]), **mt.get_metrics_hit(zzz,eee,4)}
    met = met[['RMB', 'MAE', 'RMSE', 'CC', 'POD', 'FAR', 'CSI']]
    met.to_csv(met_path)
    print(met)

plot_all_scatters(aaa, bbb, ccc, eee, zzz, f"{path}/example-hour-{date}.png", f"{path}/example-hour-{date}.csv")



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

    import cartopy.crs as ccrs
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    # plt.subplots_adjust(hspace=0.5)
    # plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(left=0, right=1.2, top=1.05, bottom=0)

    area = [116.2, 117, 39.7, 40.5]
    res = mt.RADAR(accumulation[4], 'acc', *mt.BJXSY, eles=[1.45]).ppi_wgs(0, area = area, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'CNN Model', ax=axs[0][0])
    mt.RADAR(accumulation[0], 'acc', *mt.BJXSY, eles=[1.45]).ppi_wgs(0,area = area, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'R(Z)', ax=axs[0][1])
    mt.RADAR(accumulation[1], 'acc', *mt.BJXSY, eles=[1.45]).ppi_wgs(0,area = area, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'R(kdp)', ax=axs[1][0])
    # mt.RADAR(accumulation[2], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'refzdr', save=f'{path}/example-{date}-refzdr.png')
    mt.RADAR(accumulation[3], 'acc', *mt.BJXSY, eles=[1.45]).ppi_wgs(0, area = area,scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'R(kdp,zdr)', ax=axs[1][1])

    cbar = fig.colorbar(res[0], ax = axs, location='right', pad=0.05)
    cb_tick_label = res[1].copy()
    cb_tick_label[0] = ''
    cb_tick_label[-1] = ''
    cbar.set_ticks(ticks = res[1])
    cbar.set_ticklabels(cb_tick_label)
    cbar.set_label(res[2])

    plt.savefig(f'{path}/example-{date}.png', bbox_inches='tight', dpi=fig.dpi)

if date != '':
    print(f'plot {date}')
    # path_qpe = r'/data/zry/radar/Xradar_npy_qpe/BJXSY'
    # ls = []
    # for file in os.listdir(path_qpe):
    #     if date in file:
    #         ls += [path_qpe + '/' + file]
    # accumulation = summary(ls)
    # np.save(f'./dataset/{date}.npy', accumulation)
    accumulation = np.load(f'./dataset/{date}.npy')
    compare_event(zzz, accumulation)