import sys
sys.path.append('my')
import my.point2radar as p2r

df = p2r.lookup_site2radar('/home/zry/code/gauge2bjxsy_145.csv', '/data/zry/radar/Xradar_npy_qpe/BJXSY', temp_var=0)
df.to_csv('/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-ref.csv')

df = p2r.lookup_site2radar('/home/zry/code/gauge2bjxsy_145.csv', '/data/zry/radar/Xradar_npy_qpe/BJXSY', temp_var=1)
df.to_csv('/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-kdp.csv')

df = p2r.lookup_site2radar('/home/zry/code/gauge2bjxsy_145.csv', '/data/zry/radar/Xradar_npy_qpe/BJXSY', temp_var=2)
df.to_csv('/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-refzdr.csv')

df = p2r.lookup_site2radar('/home/zry/code/gauge2bjxsy_145.csv', '/data/zry/radar/Xradar_npy_qpe/BJXSY', temp_var=3)
df.to_csv('/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-kdpzdr.csv')

# df = p2r.lookup_site2radar('/home/zry/code/gauge2bjxsy_145.csv', '/data/zry/radar/Xradar_npy_qpe/BJXSY', temp_var=4)
# df.to_csv('/home/zry/code/QPE_prv_ppi/model/based_on_202407/240727-cnn-9prv-3out-wmse/example-dl.csv')