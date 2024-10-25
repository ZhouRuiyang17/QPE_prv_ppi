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





'''scatters'''
def plot_all_scatters(aaa, bbb, ccc, ddd, eee, fff, zzz,
                      fig_path, met_path, box_path,plot=1):
    loc = (zzz>=0.1) & (aaa>=0.1) & (bbb>=0.1) & (ccc>=0.1) & (eee>=0.1) & (fff>=0.1) & (ddd>=0.1)
    
    aaa = aaa[loc].values.flatten()
    bbb = bbb[loc].values.flatten()
    ccc = ccc[loc].values.flatten()
    ddd = ddd[loc].values.flatten()
    eee = eee[loc].values.flatten()
    fff = fff[loc].values.flatten()
    zzz = zzz[loc].values.flatten()
    
    if plot:
        # 创建一个包含3个子图的图形
        fig, axs = plt.subplots(3, 2, figsize=(18, 10))
        # plt.subplots_adjust(hspace=0.5)
        # plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(left=0, right=0.9, top=1.3, bottom=0)
        
        hd1 = mt.plot_hist2d(zzz, aaa, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1,ax=axs[0][0])
        hd2 = mt.plot_hist2d(zzz, bbb, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1,ax=axs[0][1])
        hd3 = mt.plot_hist2d(zzz, ccc, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1,ax=axs[1][0])
        hd4 = mt.plot_hist2d(zzz, ddd, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1,ax=axs[1][1])
        hd5 = mt.plot_hist2d(zzz, eee, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1,ax=axs[2][0])
        hd6 = mt.plot_hist2d(zzz, fff, bins=[np.arange(101)]*2, drawline=1, labels=['Gauge (mm)', 'Radar (mm)'], equal=1, showmet=1,ax=axs[2][1])
        
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

        '''boxplot'''
        mt.plot_boxplot(zzz,[aaa,bbb,ccc,ddd,eee,fff],titles,[0,10,20,30,40,50,100],box_path)

    
    '''metrics'''
    met =                {**mt.get_metrics(zzz, aaa)}
    met = pd.DataFrame(met, index=[titles[0]], columns=met.keys())
    met.loc[titles[1]] = {**mt.get_metrics(zzz, bbb)}
    met.loc[titles[2]] = {**mt.get_metrics(zzz, ccc)}
    met.loc[titles[3]] = {**mt.get_metrics(zzz, ddd)}
    met.loc[titles[4]] = {**mt.get_metrics(zzz, eee)}
    met.loc[titles[5]] = {**mt.get_metrics(zzz, fff)}
    met = met[['RMB', 'MAE', 'RMSE', 'CC']].round(2)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = met.T.plot(kind='bar', ax=ax,width=0.8)
    for bar in bars.containers:
       ax.bar_label(bar, fontsize=8)
    ax.set_yticks(range(7))
    ax.grid()
    fig.savefig(f'{met_path}.png',bbox_inches='tight')
    # import seaborn as sns
    # met_melted = met.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='Metric', y='Value', hue='index', data=met_melted)
    # plt.savefig('./2.png')

    met.to_csv(met_path)
    # print(met)
    return met
plot_all_scatters(aaa, bbb, ccc, ddd,eee,fff, zzz, f"{path}/example-hour.png", f"{path}/example-hour.csv", f"{path}/example-hour-box.png",plot=0)



'''metrics vary with date'''
# met = pd.DataFrame(columns=['RMB', 'MAE', 'RMSE', 'CC'])
# cases = [20190517,20190722, 20190728,20190729, 20190804,20190805, 20190806, 20190809, 20190909, 20190912]
# zzzmeans = []
# for date in cases:
#     date = str(date)
#     aaa1 = aaa.loc[date]
#     bbb1 = bbb.loc[date]
#     ccc1 = ccc.loc[date]
#     ddd1 = ddd.loc[date]
#     eee1 = eee.loc[date]
#     fff1 = fff.loc[date]
#     zzz1 = zzz.loc[date]
#     met1 = plot_all_scatters(aaa1, bbb1, ccc1, ddd1,eee1,fff1, zzz1,
#                               f"{path}/example-hour-{date}.png", f"{path}/example-hour-{date}.csv", f"{path}/example-hour-box-{date}.png",
#                               plot=0)
#     met = pd.concat((met, met1))
#     zzzmeans += [zzz.loc[date].sum(axis=0).values.flatten().max()]
#     print(date, 'done')
# # print(met)
# # met.to_csv('./aaa.csv')
# fig, axs = plt.subplots(3,1,figsize=(12,10))
# for i in range(len(titles)):
#     method = titles[i]
#     line_style = '-' if 'Model' in method else '--'
#     axs[0].plot(np.arange(len(cases)), met.loc[method, 'RMB'], label=method, linestyle=line_style)
#     axs[1].plot(np.arange(len(cases)), met.loc[method, 'MAE'], label=method, linestyle=line_style)
#     axs[2].plot(np.arange(len(cases)), met.loc[method, 'RMSE'], label=method, linestyle=line_style)
#     # axs[3].plot(np.arange(len(cases)), zzzmeans)
# for i in range(3):
#     axs[i].legend()  # 显示图例
#     axs[i].grid(True)  # 显示网格
# # axs[3].set_ylabel('Max 1h rainfall (mm)')
# axs[0].set_xticks(np.arange(len(cases)),cases)
# axs[1].set_xticks(np.arange(len(cases)),cases)
# axs[2].set_xticks(np.arange(len(cases)),cases)
# axs[0].set_title('RMB')
# axs[1].set_title('MAE')
# axs[2].set_title('RMSE')
# axs[0].plot(np.arange(len(cases)), [0]*len(cases), c='black')
# plt.savefig(f"{path}/example-hour-cases.png")

