import my.utils.point2radar as p2r
import os
import pandas as pd

siteinfo_path = '/home/zry/code/gauge2bjxsy_145.csv'
path = '/data/zry/radar/Xradar_npy_qpe/run2019'
savepath = '/home/zry/code/QPE_prv_ppi/dataset/20240916eval_2models'
if not os.path.exists(savepath):
    os.makedirs(savepath)

path = '/data/zry/radar/Xradar_npy_qpe/run2019'
df = pd.DataFrame(columns=['bjxsy'])
for root, dirs, files in os.walk(path):
    for file in files:
        vec = file.split('.')
        timestamp = vec[-3] + vec[-2]

        df.loc[timestamp, 'bjxsy'] = os.path.join(root, file)
df.index = pd.to_datetime(df.index)
df.sort_index().to_csv('./run2019.csv')


# df,df1,df2,df3,df4,df5 = p2r.lookup_site2radar(siteinfo_path, path)
# df.to_csv(f'{savepath}/example-ref.csv')
# df1.to_csv(f'{savepath}/example-kdp.csv')
# df2.to_csv(f'{savepath}/example-refzdr.csv')
# df3.to_csv(f'{savepath}/example-kdpzdr.csv')
# df4.to_csv(f'{savepath}/example-res.csv')
# df5.to_csv(f'{savepath}/example-cnn.csv')