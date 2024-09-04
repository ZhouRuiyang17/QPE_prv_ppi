import sys
sys.path.append('my')
import my.utils.point2radar as p2r
import os

siteinfo_path = '/home/zry/code/gauge2bjxsy_145.csv'
path = '/data/zry/radar/Xradar_npy_qpe/BJXSY-test/ResQPE-3399-1-vlr-wmse'
savepath = '/home/zry/code/QPE_prv_ppi/model/based_on_202407/ResQPE-3399-1-vlr-wmse/eval'
if not os.path.exists(savepath):
    os.makedirs(savepath)

df = p2r.lookup_site2radar(siteinfo_path, path, temp_var=0)
df.to_csv(f'{savepath}/example-ref.csv')

df = p2r.lookup_site2radar(siteinfo_path, path, temp_var=1)
df.to_csv(f'{savepath}/example-kdp.csv')

df = p2r.lookup_site2radar(siteinfo_path, path, temp_var=2)
df.to_csv(f'{savepath}/example-refzdr.csv')

df = p2r.lookup_site2radar(siteinfo_path, path, temp_var=3)
df.to_csv(f'{savepath}/example-kdpzdr.csv')

df = p2r.lookup_site2radar(siteinfo_path, path, temp_var=4)
df.to_csv(f'{savepath}/example-dl.csv')