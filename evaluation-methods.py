import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime
import matplotlib.colors as colors
# plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.size'] = 12


aaa = mt.readcsv(r"/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-dl.csv", 0, isrr=3, mask=1, acc='H')
bbb = mt.readcsv(r"/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-ref.csv", 0, isrr=3, mask=1, acc='H')
ccc = mt.readcsv(r"/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-kdp.csv", 0, isrr=3, mask=1, acc='H')
ddd = mt.readcsv(r"/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-refzdr.csv", 0, isrr=3, mask=1, acc='H')
eee = mt.readcsv(r"/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-kdpzdr.csv", 0, isrr=3, mask=1, acc='H')
zzz = mt.readcsv(r"/home/zry/code/gauge_all.csv", 0, isrr=0, mask=1)

titles = ['dl', 'ref', 'kdp', 'refzdr', 'kdpzdr']

'''match'''
_, _, index, columns = mt.match_df(aaa, zzz)
aaa = aaa.loc[index, columns]
bbb = bbb.loc[index, columns]
ccc = ccc.loc[index, columns]
ddd = ddd.loc[index, columns]
eee = eee.loc[index, columns]
zzz = zzz.loc[index, columns]

# =============================================================================
# aaa = aaa.loc['20180716']
# bbb = bbb.loc['20180716']
# # ccc = ccc.loc['20180716']
# zzz = zzz.loc['20180716']
# =============================================================================


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
    
    hd1 = mt.hist2d(axs[0], zzz[loc], aaa[loc], bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd2 = mt.hist2d(axs[1], zzz[loc], bbb[loc], bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd3 = mt.hist2d(axs[2], zzz[loc], ccc[loc], bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd4 = mt.hist2d(axs[3], zzz[loc], ddd[loc], bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    hd5 = mt.hist2d(axs[4], zzz[loc], eee[loc], bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1)
    
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

plot_all_scatters(aaa, bbb, ccc, ddd, eee, zzz, r"model/based_on_202407/240727-cnn-9prv-3out-wmse/all.png", r"model/based_on_202407/240727-cnn-9prv-3out-wmse/all.csv")


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

plot_box(aaa, bbb, ccc, ddd, eee, zzz, titles, "model/based_on_202407/240727-cnn-9prv-3out-wmse/all-box.png", "model/based_on_202407/240727-cnn-9prv-3out-wmse/all-hist.png")


# =============================================================================
# def accumulate(ls):
#     print(f'{ls[0]}-{ls[-1]}')
#     
#     data = np.load(ls[0])
#     for fpath in ls[1:]:
#         temp = np.load(fpath)
#         data += temp
#     
#     return data
# 
# def plot_an_event(aaa, bbb, ccc, zzz, event,
#                   ls1, ls2, ls3,
#                   siteinfo_path,
#                   savepath):
#     aaa = aaa.loc[event[0] : event[1]].copy()
#     bbb = bbb.loc[event[0] : event[1]].copy()
#     ccc = ccc.loc[event[0] : event[1]].copy()
#     zzz = zzz.loc[event[0] : event[1]].copy()
#     
#     plot_all_scatters(aaa,bbb,ccc,zzz, savepath[0], savepath[1])
#     
#     columns = zzz.columns
#     aaa = aaa.sum().values
#     bbb = bbb.sum().values
#     ccc = ccc.sum().values
#     zzz = zzz.sum().values
#     loc = (zzz>=2) & (aaa>=2) & (bbb>=2) & (ccc>=2)
# 
#     
#     # 创建一个包含3个子图的图形
#     import cartopy.crs as ccrs
#     fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': ccrs.PlateCarree()})
#     # plt.subplots_adjust(hspace=0.5)
#     # plt.subplots_adjust(wspace=0.5)
#     plt.subplots_adjust(left=0, right=1.2, top=1, bottom=0)
#     
#     area, lon, lat = mt.yinchuan_c_vcp()
#     siteinfo = pd.read_csv(siteinfo_path, index_col=0)
#     siteinfo.index = siteinfo.index.astype(str)
#     lons = siteinfo.loc[columns, 'lon']
#     lats = siteinfo.loc[columns, 'lat']
#     
#     pm1 = mt.pcolor_geo(axs[0], accumulate(ls1), 'acc', area, lon, lat, scatters=[lons[loc], lats[loc], zzz[loc]], size=50)
#     pm2 = mt.pcolor_geo(axs[1], accumulate(ls2), 'acc', area, lon, lat, scatters=[lons[loc], lats[loc], zzz[loc]], size=50)
#     pm3 = mt.pcolor_geo(axs[2], accumulate(ls3), 'acc', area, lon, lat, scatters=[lons[loc], lats[loc], zzz[loc]], size=50)
# 
#     
#     cbar = fig.colorbar(pm1[0], ax=axs, location='right')
#     cbar.set_ticks(ticks = pm1[2])
#     cbar.set_ticklabels(pm1[2])
#     cbar.set_label(pm1[1])
#     fig.savefig(savepath[2], transparent=0, dpi=fig.dpi, bbox_inches='tight')
#     
# 
# 
# '''20180720'''
# path = r'D:\data\yinchuan\mosaic\20240521结果\zr300'
# ls = []; ls2=[]; ls3=[]
# for file in os.listdir(path):
#     if '20180720' in file:# or '20220711' in file:
#         ls += [path + '\\' + file]
#         ls2 += [path.replace('zr300', 'mlp') + '\\' + file]
#         ls3 += [path.replace('zr300', 'mlpyc') + '\\' + file]
# begin = '2018-7-20 9:0'#!!!
# end = '2018-7-21 0:0'
# print(ls[170])
# ls = ls[:170]
# savepath = [r"C:\Users\admin\OneDrive\讨论\银川内涝\文章\银川\new\yc-20180720.png", 
#             r"C:\Users\admin\OneDrive\讨论\银川内涝\文章\银川\new\yc-20180720.csv",
#             r"C:\Users\admin\OneDrive\讨论\银川内涝\文章\银川\new\yc-20180720-2.png"]
# plot_an_event(aaa,bbb,ccc,zzz,[begin, end], ls, ls, ls, r"D:\data\yinchuan\gauge\site_info.csv", savepath)
# 
# '''20180722'''
# path = r'D:\data\yinchuan\mosaic\20240521结果\zr300'
# ls = []; ls2=[]; ls3=[]
# for file in os.listdir(path):
#     if '20180722' in file:# or '20220711' in file:
#         ls += [path + '\\' + file]
#         ls2 += [path.replace('zr300', 'mlp') + '\\' + file]
#         ls3 += [path.replace('zr300', 'mlpyc') + '\\' + file]
# begin = '2018-7-22 9:0'#!!!
# end = '2018-7-23 8:0'
# print(ls[254])
# ls = ls[:255]
# savepath = [r"C:\Users\admin\OneDrive\讨论\银川内涝\文章\银川\new\yc-20180722.png", 
#             r"C:\Users\admin\OneDrive\讨论\银川内涝\文章\银川\new\yc-20180722.csv",
#             r"C:\Users\admin\OneDrive\讨论\银川内涝\文章\银川\new\yc-20180722-2.png"]
# plot_an_event(aaa,bbb,ccc,zzz,[begin, end], ls, ls2, ls3, r"D:\data\yinchuan\gauge\site_info.csv", savepath)
# 
# =============================================================================
