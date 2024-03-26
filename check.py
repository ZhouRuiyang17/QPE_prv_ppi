import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import my.mytools as mt
import datetime

'''load dataset'''
dataset = np.load(r"D:\data\dataset\prv_ppi\raw\dataset.npy")
index = pd.read_csv(r"D:\data\dataset\prv_ppi\raw\index.csv", index_col = 0)
idx = list(set(index['0']))
clm = list(set(index['1']))

ref = dataset[:,5,4,4]
df = pd.DataFrame(columns = clm, index = idx)
for i in range(len(ref)):
    ts = index.iloc[i,0]
    st = index.iloc[i,1]
    df.loc[ts, st] = ref[i]
    print(ts, st)
    # break
df.to_csv(r"D:\data\dataset\prv_ppi\dataset20240325\kdp_zzqc.csv")
#%%
'''1h'''
prv = pd.read_csv(r"D:\data\dataset\prv_ppi\dataset20240325\kdp_zzqc.csv", index_col = 0).fillna(-999)
prv.index = pd.to_datetime(prv.index)
prv[prv<0] = 0
# prv = 10**(prv/10)
# a,b = 0.0576, 0.557
# a,b = 15.421, 0.817
# rainrate = a*prv**b
zdr = pd.read_csv(r"D:\data\dataset\prv_ppi\dataset20240325\zdr_zzqc.csv", index_col = 0).fillna(-999)
zdr.index = pd.to_datetime(zdr.index)
zdr[zdr<0] = 0
zdr = 10**(zdr/10)
# a,b,c = 0.0059, 0.994, -4.929
a,b,c = 26.778, 0.946, -1.249
rainrate = a*prv**b*zdr**c
rainrate[rainrate < 0.1] = 0
acc3min = rainrate * 3/60
acc1h = acc3min.resample('H', label = 'right').sum()

gauge = pd.read_csv(r"D:\data\beijing\gauge\城市院自动站_all.csv", index_col = 0).fillna(0)
gauge.index = pd.to_datetime(gauge.index)
gauge[gauge<0] = 0
gauge[gauge>1000] = 0


index_common = list(set(acc1h.index)&set(gauge.index))
columns = acc1h.columns
acc1h = acc1h.loc[index_common, columns]
gauge = gauge.loc[index_common, columns]

xxx = gauge.values.flatten()
yyy = acc1h.values.flatten()
loc = (xxx>0) & (yyy>0)
scatter = mt.Scatter(xxx[loc], yyy[loc])
scatter.plot3(bins = [np.arange(0, 50, 0.5)]*2, lim=[[0,50]]*2, show_metrics=1, draw_line=1)

'''RR'''
# prv = pd.read_csv(r"D:\data\ref_zhangzheQC_dataset.csv", index_col = 0).fillna(-999)
# prv.index = pd.to_datetime(prv.index)
# prv[prv<0] = 0
# prv = 10**(prv/10)
# a,b = 0.03468, 0.5869
# # a,b = 14.93, 0.83
# rainrate = a*prv**b
# # =============================================================================
# # zdr = pd.read_csv(r"D:\data\zdr_zhangzheQC_dataset.csv", index_col = 0).fillna(-999)
# # zdr.index = pd.to_datetime(zdr.index)
# # zdr[zdr<0] = 0
# # zdr = 10**(zdr/10)
# # # a,b,c = 0.00614, 0.959, -3.671
# # a,b,c = 22.56, 0.91, -0.859
# # rainrate = a*prv**b*zdr**c
# # =============================================================================
# rainrate[rainrate < 0.1] = 0


# gauge = pd.read_csv(r"D:\data\RR_all.csv", index_col = 0).fillna(0)
# gauge.index = pd.to_datetime(gauge.index)
# gauge[gauge<0] = 0
# gauge[gauge>1000] = 0
# gauge = gauge.resample('3min', label='left').mean()

# index_radar = rainrate.index.to_frame()
# index_gauge = gauge.index.to_frame()
# index_common = pd.merge(index_gauge, index_radar)
# columns = rainrate.columns
# rainrate = rainrate.loc[index_common[0], columns]
# gauge = gauge.loc[index_common[0], columns]

# xxx = gauge.values.flatten()
# yyy = rainrate.values.flatten()
# loc = (xxx>0) & (yyy>0)
# scatter = mt.Scatter(xxx[loc], yyy[loc])
# scatter.plot3(bins = [np.arange(0, 50, 0.5)]*2, lim=[[0,50]]*2, show_metrics=1, draw_line=1)
