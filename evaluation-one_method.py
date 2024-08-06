import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.utils.mytools as mt
import datetime

#%%
pathsave = '/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse'
# import gauge and radar: acc
radar = mt.readcsv(r"/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-dl.csv", 0, isrr=3, mask=1, acc='H')
gauge = mt.readcsv(r"/home/zry/code/gauge_all.csv", 0, isrr=0, mask=1, acc='no')

# match sites and timestamp
columns = list(set(radar.columns)&set(gauge.columns))
index = list(set(radar.index)&set(gauge.index))
radar = radar.loc[index, columns].sort_index()
gauge = gauge.loc[index, columns].sort_index()


# # 要删除的日期列表
# dates_to_remove = ['20170521','20170522','20170523',
#                    '20170810',
#                    '20190525','20190526','20190527']
# # 将日期转换为datetime对象
# dates_to_remove = pd.to_datetime(dates_to_remove)
# # 过滤掉要删除的日期
# radar = radar[~radar.index.normalize().isin(dates_to_remove)]
# gauge = gauge[~gauge.index.normalize().isin(dates_to_remove)]


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


    mt.RADAR(accumulation[4], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'dl', save=f'{pathsave}/example-dl.png')
    mt.RADAR(accumulation[0], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'ref', save=f'{pathsave}/example-ref.png')
    mt.RADAR(accumulation[1], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'kdp', save=f'{pathsave}/example-kdp.png')
    mt.RADAR(accumulation[2], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'refzdr', save=f'{pathsave}/example-refzdr.png')
    mt.RADAR(accumulation[3], 'acc', *mt.BJXSY, eles=[1.45]).ppi_lonlat(0, scatters=[lons[loc], lats[loc], gaugesum[loc]], title = f'kdpzdr', save=f'{pathsave}/example-kdpzdr.png')

path = r'/data/zry/radar/Xradar_npy_qpe/BJXSY'
ls = []
for file in os.listdir(path):
        ls += [path + '/' + file]
compare_event(gauge, summary(ls))


