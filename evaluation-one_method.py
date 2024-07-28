import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime

#%%

# import gauge and radar: acc
radar = mt.readcsv(r"/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-kdpzdr.csv", 0, isrr=3, mask=1, acc='H')
gauge = mt.readcsv(r"/home/zry/code/gauge_all.csv", 0, isrr=0, mask=1, acc='no')

# match sites and timestamp
columns = list(set(radar.columns)&set(gauge.columns))
index = list(set(radar.index)&set(gauge.index))
radar = radar.loc[index, columns].sort_index()
gauge = gauge.loc[index, columns].sort_index()


# 要删除的日期列表
dates_to_remove = ['20170521','20170522','20170523',
                   '20170810',
                   '20190525','20190526','20190527']
# 将日期转换为datetime对象
dates_to_remove = pd.to_datetime(dates_to_remove)
# 过滤掉要删除的日期
radar = radar[~radar.index.normalize().isin(dates_to_remove)]
gauge = gauge[~gauge.index.normalize().isin(dates_to_remove)]


# date = '20170822'
# radar = radar.loc[date]
# gauge = gauge.loc[date]

# site = '54419'
# mt.plot2l([radar[site], gauge[site]], ['Rdar', 'Gage'], '')



#%%

# for date in sorted(os.listdir(r'G:\mosaic_npz_5min\qpe_acc5min')):
    
    
def summary(ls):
    count = 0
    lenth = len(ls)
    for fp in ls:
        if count == 0:
            data = np.load(fp)['griddata']
            count += 1
        else:
            data += np.load(fp)['griddata']
            count += 1

    print(f'{count} / {lenth}')
    return data

def compare_event(radar, gauge, event, subtitle, accumulation):
    '''clip by date'''
    if len(event) == 2:
        radar = radar.loc[event[0] : event[1]].copy()
        gauge = gauge.loc[event[0] : event[1]].copy()
        title = '{}-{}'.format(event[0],event[1]) 
    else:
        radar = radar.loc[event].copy()
        gauge = gauge.loc[event].copy()
        title = '{}'.format(event)

    if radar.columns.all() == gauge.columns.all():
        print('yes!')

    '''1h scatter'''
    radar1h = radar.values.flatten()
    gauge1h = gauge.values.flatten()
    loc = (radar1h >= 0.1) & (gauge1h >= 0.1)
    scatter = mt.Scatter(gauge1h[loc], radar1h[loc])
    scatter.plot3(bins = [np.arange(100)]*2,
                  label = ['1h rainfall (mm) (gauge)', '1h rainfall (mm) (radar)'], lim = [[0,100],[0,100]],
                  show_metrics=1,draw_line=1, title = title + ' ' + subtitle)
        
    '''acc scatter'''
    sites_used = gauge.columns
    radarsum = radar.sum()
    gaugesum = gauge.sum()
    loc = (radarsum > 10) & (gaugesum > 10)

    




    '''site info'''
    siteinfo = pd.read_csv(r"D:\data\beijing\site_info\城市院自动站_position.csv",index_col=0).sort_values(by='stnm')
    lons = siteinfo.loc[sites_used, 'lon']
    lats = siteinfo.loc[sites_used, 'lat']

    grid_lon, grid_lat = mt.beijing_x_500()
    # lon = lon[300:500]
    # lat = lat[350:550]
    # area = [115.4, 117.6, 39.4, 41.1]
    mt.plot_grid(accumulation, 'acc', mt.BJ_AREA, grid_lon, grid_lat, scatters=[lons[loc], lats[loc], gaugesum[loc]], size=50, figpath=r'C:\Users\admin\OneDrive\图片\86天的单日累计\待散点\{}.png'.format(date))
    # mt.plot_grid_diff(accumulation, 'acc', mt.BJ_AREA, grid_lon, grid_lat, scatters=[lons[loc], lats[loc], radarsum[loc] - gaugesum[loc]], size=50)



    # mt.RADAR(data, 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'{title}+{subtitle}')

# path = r''
# ls = []
# for root, dirs, files in os.walk(path):
#     for file in sorted(files):
#         if date in file and '5min' in file:
#             ls += [root + '\\' + file]
# compare_event(radar, gauge, date, '', summary(ls))



     
# ----all: 1h-rainfall
radar = radar.values.flatten()
gauge = gauge.values.flatten()

loc = (radar >= 0.1) & (gauge >= 0.1)
scatter = mt.Scatter(gauge[loc], radar[loc])
scatter.plot3(bins = [np.arange(0,151,1)]*2, labels = ['gauge (mm)', 'radar (mm)'], lim = [[0,151],[0,151]], show_metrics=1,draw_line=1, title = '', fit=0, fpath='/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-kdpzdr.png')

# delta = radar[loc] - gauge[loc]
# plt.boxplot(delta, showfliers=1);plt.show()
# plt.boxplot(delta, showfliers=0);plt.show()
# plt.hist(delta,bins = np.arange(-100,90,2));plt.show()

# uncor = radar[loc]
# true = gauge[loc]
# kkk = true.mean() / uncor.mean()
# print(kkk)
#%%



# =============================================================================
# def function(uncor, k, c, a):
#     return k*uncor + c + a*uncor**2
# from scipy.optimize import curve_fit
# para = curve_fit(function, uncor, true)
# cor = function(uncor, *para[0])
# scatter = mt.Scatter(true, cor)
# scatter.plot3(bins = [np.arange(0,151,1)]*2, label = ['gauge (mm)', 'radar (mm)'], lim = [[0,151],[0,151]], show_metrics=1,draw_line=1, title = '', fit=0)
# =============================================================================
# =============================================================================
# uncor = np.arange(100)
# plt.plot(uncor, uncor)
# plt.plot(uncor, function(uncor, *para[0]))
# plt.plot(uncor, uncor*kkk)
# =============================================================================
# scatter = mt.Scatter(true, uncor*1.2)
# scatter.plot3(bins = [np.arange(0,151,1)]*2, label = ['gauge (mm)', 'radar (mm)'], lim = [[0,151],[0,151]], show_metrics=1,draw_line=1, title = '', fit=0)
