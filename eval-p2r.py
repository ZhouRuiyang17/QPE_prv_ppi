import sys
sys.path.append('my')
import my.utils.point2radar as p2r


siteinfo_path = '/home/zry/code/gauge2bjxsy_145.csv'
path = '/data/zry/radar/Xradar_npy_qpe/BJXSY'
savepath = '/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/'

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