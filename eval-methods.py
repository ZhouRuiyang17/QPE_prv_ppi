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
_, _, index, columns = mt.match_df(aaa, zzz)
aaa = aaa.loc[index, columns]
bbb = bbb.loc[index, columns]
ccc = ccc.loc[index, columns]
ddd = ddd.loc[index, columns]
eee = eee.loc[index, columns]
fff = fff.loc[index, columns]
zzz = zzz.loc[index, columns]





'''scatters'''
def plot_all_scatters(aaa, bbb, ccc, ddd, eee, fff, zzz,
                      fig_path, met_path, box_path,):
    loc = (zzz>=0.1) & (aaa>=0.1) & (bbb>=0.1) & (ccc>=0.1) & (eee>=0.1) & (fff>=0.1) & (ddd>=0.1)
    
    aaa = aaa[loc].values.flatten()
    bbb = bbb[loc].values.flatten()
    ccc = ccc[loc].values.flatten()
    ddd = ddd[loc].values.flatten()
    eee = eee[loc].values.flatten()
    fff = fff[loc].values.flatten()
    zzz = zzz[loc].values.flatten()
    
    # 创建一个包含3个子图的图形
    fig, axs = plt.subplots(3, 2, figsize=(18, 10))
    # plt.subplots_adjust(hspace=0.5)
    # plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(left=0, right=0.8, top=1.2, bottom=0)
    
    hd1 = mt.plot_hist2d(axs[0][0], zzz, aaa, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd2 = mt.plot_hist2d(axs[0][1], zzz, bbb, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd3 = mt.plot_hist2d(axs[1][0], zzz, ccc, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd4 = mt.plot_hist2d(axs[1][1], zzz, ddd, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd5 = mt.plot_hist2d(axs[2][0], zzz, eee, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd6 = mt.plot_hist2d(axs[2][1], zzz, fff, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    
    '''cb'''
    fig.colorbar(hd1[3], ax=axs, location='right', pad=0.05)
    '''title'''
    count = 0
    for i in range(3):
        for j in range(2):
            axs[i][j].set_title(titles[count])
            count += 1
    '''save'''
    fig.savefig(fig_path, transparent=0, dpi=600, bbox_inches='tight')

    # '''boxplot'''
    mt.plot_boxplot(zzz,[aaa,bbb,ccc,ddd,eee,fff],titles,[0,10,20,30,40,50,100],box_path)
    # mt.plot_boxplot(zzz,[aaa,bbb,ccc,ddd,eee,fff],titles,[0,10,20,30,40,50,100],box_path+'fly.png',1)

    
    '''metrics'''
    met =                {**mt.get_metrics(zzz, aaa)}
    met = pd.DataFrame(met, index=[titles[0]], columns=met.keys())
    met.loc[titles[1]] = {**mt.get_metrics(zzz, bbb)}
    met.loc[titles[2]] = {**mt.get_metrics(zzz, ccc)}
    met.loc[titles[3]] = {**mt.get_metrics(zzz, ddd)}
    met.loc[titles[4]] = {**mt.get_metrics(zzz, eee)}
    met.loc[titles[5]] = {**mt.get_metrics(zzz, fff)}
    met = met[['BIAS', 'MAE', 'RMSE', 'CC']]
    # met.to_csv(met_path)
    # print(met)
    return met



plot_all_scatters(aaa, bbb, ccc, ddd,eee,fff, zzz, f"{path}/example-hour.png", f"{path}/example-hour.csv", f"{path}/example-hour-box.png")

'''metrics vary with date'''
met = pd.DataFrame(columns=['BIAS', 'MAE', 'RMSE', 'CC'])
cases = [20190517, 20190722, 20190729, 20190804, 20190806, 20190809, 20190909, 20190912]
zzzmeans = []
for date in cases:
    date = str(date)
    aaa1 = aaa.loc[date]
    bbb1 = bbb.loc[date]
    ccc1 = ccc.loc[date]
    ddd1 = ddd.loc[date]
    eee1 = eee.loc[date]
    fff1 = fff.loc[date]
    zzz1 = zzz.loc[date]
    met1 = plot_all_scatters(aaa1, bbb1, ccc1, ddd1,eee1,fff1, zzz1, f"{path}/example-hour-{date}.png", f"{path}/example-hour-{date}.csv", f"{path}/example-hour-box-{date}.png")
    met = pd.concat((met, met1))
    zzzmeans += [zzz.loc[date].sum(axis=0).values.flatten().mean()]
    print(date, 'done')
print(met)
fig, axs = plt.subplots(4,1,figsize=(12,10))
for i in range(len(titles)):
    method = titles[i]
    line_style = '-' if 'Model' in method else '--'
    axs[0].plot(np.arange(len(cases)), met.loc[method, 'BIAS'], label=method, linestyle=line_style)
    axs[1].plot(np.arange(len(cases)), met.loc[method, 'MAE'], label=method, linestyle=line_style)
    axs[2].plot(np.arange(len(cases)), met.loc[method, 'RMSE'], label=method, linestyle=line_style)
    axs[3].plot(np.arange(len(cases)), zzzmeans)
for i in range(3):
    axs[i].legend()  # 显示图例
    axs[i].grid(True)  # 显示网格
plt.savefig(f"{path}/example-hour-cases.png")


date = '20190804'
'''timeseries'''
def plot_timeseries():
    zzz1 = zzz
    aaa1 = aaa
    bbb1 = bbb
    ccc1 = ccc
    ddd1 = ddd
    eee1 = eee
    fff1 = fff

    loc = (zzz1>=0.1) & (aaa1>=0.1) & (bbb1>=0.1) & (ccc1>=0.1) & (ddd1>=0.1) & (eee1>=0.1) & (fff1>=0.1)
    zzz1 = zzz1.where(loc, np.nan)
    aaa1 = aaa1.where(loc, np.nan)
    bbb1 = bbb1.where(loc, np.nan)
    ccc1 = ccc1.where(loc, np.nan)
    ddd1 = ddd1.where(loc, np.nan)
    eee1 = eee1.where(loc, np.nan)
    fff1 = fff1.where(loc, np.nan)

    zzz1 = zzz1.mean(axis=1)
    aaa1 = aaa1.mean(axis=1)
    bbb1 = bbb1.mean(axis=1)
    ccc1 = ccc1.mean(axis=1)
    ddd1 = ddd1.mean(axis=1)
    eee1 = eee1.mean(axis=1)
    fff1 = fff1.mean(axis=1)

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
    plt.savefig(f'{path}/{date}.png')


# plot_timeseries()




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

# if date != '':
#     print(f'plot {date}')
#     path_qpe = r'/data/zry/radar/Xradar_npy_qpe/BJXSY-test/ResQPE-3399-1-vlr-wmse'
#     ls = []
#     for file in os.listdir(path_qpe):
#         if date in file:
#             ls += [path_qpe + '/' + file]
#     accumulation = summary(ls)
#     # np.save(f'./dataset/{date}.npy', accumulation)
#     # accumulation = np.load(f'./dataset/{date}-1hour.npy')
#     compare_event(zzz, accumulation)