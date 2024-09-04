import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import my.utils.utils_ml as utils
import my.utils.mytools as mt
from common_tool import *

import datetime
import logging
import psutil


def fast_test(path_save, model, device, nettype='res'):
    count = 0
    for file in os.listdir('../'):
        if file.endswith('npz') and '2019' in file:
            print(file)
            f = np.load(f'../{file}')
            # data = f['data'][:, [3,4,5], 30-4:30+4+1, 30-4:30+4+1]
            data = f['data'][:, :, 30-4:30+4+1, 30-4:30+4+1]
            ts = [datetime.datetime.strptime(ts, '%Y%m%d%H%M') for ts in f['ts']]

            if count == 0:
                radar = pd.DataFrame(index=ts)
                radar1 = radar.copy()
                radar2 = radar.copy()
                radar3 = radar.copy()
                radar4 = radar.copy()
                count += 1
            
            stnm = file[5:10]
            if nettype == 'cnn':
               radar.loc[ts, stnm] = apply_CNNQPE(data,model,device)
            elif nettype == 'res':
                radar.loc[ts, stnm] = apply_ResQPE(data,model,device)
            # radar.loc[ts, stnm] = apply_resver3(data,model,device)

            rr1, rr2, rr3, rr4 = qpe_3ele(data[:,3,4,4],data[:,4,4,4],data[:,5,4,4])
            radar1.loc[ts, stnm] = rr1
            radar2.loc[ts, stnm] = rr2
            radar3.loc[ts, stnm] = rr3
            radar4.loc[ts, stnm] = rr4
            # break

    radar.to_csv(f'{path_save}/test-rr-dl.csv')
    radar1.to_csv(f'{path_save}/test-rr-ref.csv')
    radar2.to_csv(f'{path_save}/test-rr-kdp.csv')
    radar3.to_csv(f'{path_save}/test-rr-refzdr.csv')
    radar4.to_csv(f'{path_save}/test-rr-kdpzdr.csv')

    radar = mt.readcsv(f'{path_save}/test-rr-dl.csv', isrr=3, mask=1, acc='H')
    radar1 = mt.readcsv(f'{path_save}/test-rr-ref.csv', isrr=3, mask=1, acc='H')
    radar2 = mt.readcsv(f'{path_save}/test-rr-kdp.csv', isrr=3, mask=1, acc='H')
    radar3 = mt.readcsv(f'{path_save}/test-rr-refzdr.csv', isrr=3, mask=1, acc='H')
    radar4 = mt.readcsv(f'{path_save}/test-rr-kdpzdr.csv', isrr=3, mask=1, acc='H')
    gauge = mt.readcsv('../gauge_all.csv')

    _, _, idx, col = mt.match_df(gauge, radar)
    radar = radar.loc[idx, col]
    radar1 = radar1.loc[idx, col]
    radar2 = radar2.loc[idx, col]
    radar3 = radar3.loc[idx, col]
    radar4 = radar4.loc[idx, col]
    gauge = gauge.loc[idx, col]

    zzz = gauge.values
    aaa = radar.values
    bbb = radar1.values
    ccc = radar2.values
    ddd = radar3.values
    eee = radar4.values
    loc = (zzz>=0.1) & (aaa>=0.1) & (bbb>=0.1) & (ccc>=0.1) & (ddd>=0.1) & (eee>=0.1)
    mt.Scatter(zzz[loc], aaa[loc]).plot3(bins=[np.arange(60)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,60]]*2, show_metrics=1, draw_line=1,
                                          fpath=f'{path_save}/test-hour-dl.png', title='dl')
    mt.Scatter(zzz[loc], bbb[loc]).plot3(bins=[np.arange(60)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,60]]*2, show_metrics=1, draw_line=1,
                                        fpath=f'{path_save}/test-hour-ref.png', title='ref')
    mt.Scatter(zzz[loc], ccc[loc]).plot3(bins=[np.arange(60)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,60]]*2, show_metrics=1, draw_line=1,
                                        fpath=f'{path_save}/test-hour-kdp.png', title='kdp')
    mt.Scatter(zzz[loc], ddd[loc]).plot3(bins=[np.arange(60)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,60]]*2, show_metrics=1, draw_line=1,
                                        fpath=f'{path_save}/test-hour-refzdr.png', title='refzdr')
    mt.Scatter(zzz[loc], eee[loc]).plot3(bins=[np.arange(60)]*2, labels=['gauge (mm)', 'radar (mm)'], lim=[[0,60]]*2, show_metrics=1, draw_line=1,
                                        fpath=f'{path_save}/test-hour-kdpzdr.png', title='kdpzdr')
    
    # mytable(zzz,aaa,bbb,ccc,ddd,eee,path_save,2)
    # mytable(zzz,aaa,bbb,ccc,ddd,eee,path_save,4)
    # mytable(zzz,aaa,bbb,ccc,ddd,eee,path_save,8)


    # plt.figure()
    # plt.boxplot([aaa[loc]-zzz[loc],
    #              bbb[loc]-zzz[loc],
    #              ccc[loc]-zzz[loc],
    #              ddd[loc]-zzz[loc],
    #              eee[loc]-zzz[loc]],
    #              labels=['dl', 'ref', 'kdp', 'refzdr', 'kdpzdr'], showfliers=1, showmeans=1)
    # plt.grid()
    # plt.savefig(f'{path_save}/test-box-fly.png')

    # plt.figure()
    # plt.boxplot([aaa[loc]-zzz[loc],
    #             bbb[loc]-zzz[loc],
    #             ccc[loc]-zzz[loc],
    #             ddd[loc]-zzz[loc],
    #             eee[loc]-zzz[loc]],
    #             labels=['dl', 'ref', 'kdp', 'refzdr', 'kdpzdr'], showfliers=0, showmeans=1)
    # plt.grid()
    # plt.savefig(f'{path_save}/test-box.png')

def example(extent = 4):
    ls = []
    lssave = []
    for root, dirs, files in os.walk('/data/zry/radar/Xradar_npz_qc/BJXSY/20190809'):
        for file in files:
            # if 'BJXSY.20190722.142959' in file:
                ls += [root + '/' + file]
                lssave += ['/data/zry/radar/Xradar_npy_qpe/BJXSY' + '/' + file.replace('npz', 'npy')]

    for fp, fpsave in zip(ls, lssave):
        logging.info(fp)
        t0 = datetime.datetime.now()
        data = np.load(fp)['data']
        logging.info(f"cost of np.load(fp)['data']: {datetime.datetime.now()-t0}")
        aaaa = data[[0,1,3], :3] # [0,1,3]表示ref zdr kdp，[:3]表示前三层，格式(prv, ele, azi, gate)

        bbbb = np.zeros((9,360,1000))*1. # (ele*prv, azi, gate)
        bbbb[[0,3,6]] = aaaa[0]
        bbbb[[1,4,7]] = aaaa[1]
        bbbb[[2,5,8]] = aaaa[2]

        cccc = np.zeros((9,360+8,1000))*1. # (ele*prv, azi_extent, gate)
        cccc[:,extent:-extent] = bbbb
        cccc[:,:extent] = bbbb[:,-extent:]
        cccc[:,-extent:] = bbbb[:,:extent]

        t0 = datetime.datetime.now()
        samples = np.zeros((360*(1000-2*extent), 9, 2*extent+1, 2*extent+1))
        counter = 0
        for true_azi in np.arange(360):
            fake_azi = true_azi+extent
            for gate in np.arange(extent,1000-extent):
                sample = cccc[:,fake_azi-extent:fake_azi+extent+1, gate-extent:gate+extent+1]
                samples[counter] = sample; counter += 1
        logging.info(f'cost of resample: {datetime.datetime.now()-t0}')
        logging.info(f'num of samples:{counter}')
        
        rr1, rr2, rr3, rr4 = qpe_3ele(samples, extent)
        rainrate = np.zeros((9,360,1000))
        rainrate[0, :, 4:-4] = rr1.reshape(360, 1000-2*extent)
        rainrate[1, :, 4:-4] = rr2.reshape(360, 1000-2*extent)
        rainrate[2, :, 4:-4] = rr3.reshape(360, 1000-2*extent)
        rainrate[3, :, 4:-4] = rr4.reshape(360, 1000-2*extent)

        t0 = datetime.datetime.now()
        rr_dl = apply_3ele(samples)
        logging.info(f'cost of qpe: {datetime.datetime.now()-t0}')
        rainrate[4, :, 4:-4] = rr_dl.reshape(360, 1000-2*extent)

        np.save(fpsave, rainrate)
        logging.info(fpsave)
        logging.info('-----------------------------------------')
        

def reshape_999(prvs, extent = 4):
    bbbb = np.zeros((9,360,1000))*1. # (ele*prv, azi, gate)
    bbbb[[0,3,6]] = prvs[0]
    bbbb[[1,4,7]] = prvs[1]
    bbbb[[2,5,8]] = prvs[2]

    cccc = np.zeros((9,360+8,1000))*1. # (ele*prv, azi_extent, gate)
    cccc[:,extent:-extent] = bbbb
    cccc[:,:extent] = bbbb[:,-extent:]
    cccc[:,-extent:] = bbbb[:,:extent]
    return cccc

def reshape_3399(data, extent = 4):
    new = np.zeros((3,3,360+2*extent,1000))

    new[:,:,extent:-extent] = data
    new[:,:,:extent] = data[:,:,-extent:]
    new[:,:,-extent:] = data[:,:,:extent]
    return new

def test(path_save, model, device):
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    df = pd.read_csv('/home/zry/code/QPE_prv_ppi/my/统计2019年的降雨-最大24小时-细化.csv')

    ls = []
    lssave = []
    for fp in df['fp']:
        ls += [fp]

        file = fp.split('/')[-1]
        fpsave = path_save + '/' + file.replace('npz', 'npy')
        lssave += [fpsave]

    for fp, fpsave in zip(ls, lssave):
        '''简单测试用'''
        # fp = '/data/zry/radar/Xradar_npz_qc/BJXSY/20180716/BJXSY.20180716.003600.npz'
        # fpsave = '/data/zry/radar/Xradar_npy_qpe/BJXSY-test/ResQPE-3399-1-vlr-wmse/BJXSY.20180716.003600.npz'
        if os.path.exists(fpsave):
            continue

        logging.info(fp)
        t0 = datetime.datetime.now()
        data = np.load(fp)['data']
        logging.info(f"cost of np.load(fp)['data']: {datetime.datetime.now()-t0}")
        prvs = data[[0,1,3], :3] # [0,1,3]表示ref zdr kdp，[:3]表示前三层，格式(prv, ele, azi, gate)

        prvs_extent = reshape_3399(prvs)

        t0 = datetime.datetime.now()
        samples = np.zeros((360*(1000-2*4), 3,3,9,9))# num_samples, 3,3,9,9
        counter = 0
        for true_azi in np.arange(360):
            fake_azi = true_azi+4
            for gate in np.arange(4,1000-4):
                sample = prvs_extent[:,:, fake_azi-4:fake_azi+4+1, gate-4:gate+4+1]
                samples[counter] = sample; counter += 1
                # print(true_azi, gate)
        logging.info(f'cost of resample: {datetime.datetime.now()-t0}')
        
        rr1, rr2, rr3, rr4 = qpe_3ele(samples[:,0,1,4,4],samples[:,1,1,4,4],samples[:,2,1,4,4])
        rainrate = np.zeros((9,360,1000))
        rainrate[0, :, 4:-4] = rr1.reshape(360, 1000-2*4)
        rainrate[1, :, 4:-4] = rr2.reshape(360, 1000-2*4)
        rainrate[2, :, 4:-4] = rr3.reshape(360, 1000-2*4)
        rainrate[3, :, 4:-4] = rr4.reshape(360, 1000-2*4)

        rr_dl = apply_ResQPE(samples, model, device, mode='test')
        rainrate[4, :, 4:-4] = rr_dl.reshape(360, 1000-2*4)

        np.save(fpsave, rainrate)
        logging.info(fpsave)

        # 在循环结束时输出内存占用情况
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logging.info(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")
        # 输出当前显存占用情况
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        logging.info(f"CUDA memory allocated: {allocated_memory / 1024 ** 2:.2f} MB")
        logging.info(f"CUDA memory reserved: {reserved_memory / 1024 ** 2:.2f} MB")
        # 清理显存
        torch.cuda.empty_cache()
        logging.info('-----------------------------------------')
    logging.info('Finished!!!!!')

def run2019(path_save, resnet, cnnnet, device):
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    '''统计文件'''
    ls = []
    lssave = []
    for root, dirs, files in os.walk('/data/zry/radar/Xradar_npz_qc/BJXSY'):
            if '2019' in root:
                for file in files:
                    ls += [f'{root}/{file}']
                    fpsave = path_save + '/' + file.replace('npz', 'npy')
                    lssave += [fpsave]

    for fp, fpsave in zip(ls, lssave):
        '''简单测试用'''
        # fp = '/data/zry/radar/Xradar_npz_qc/BJXSY/20180716/BJXSY.20180716.003600.npz'
        # fpsave = '/data/zry/radar/Xradar_npy_qpe/run2019/BJXSY.20180716.003600.npz'
        if os.path.exists(fpsave):
            continue

        '''载入数据'''
        logging.info(fp)
        t0 = datetime.datetime.now()
        data = np.load(fp)['data']
        logging.info(f"cost of np.load(fp)['data']: {datetime.datetime.now()-t0}")
        prvs = data[[0,1,3], :3] # [0,1,3]表示ref zdr kdp，[:3]表示前三层，格式(prv, ele, azi, gate)

        '''paras'''
        rainrate = np.zeros((9,360,1000))
        rr1, rr2, rr3, rr4 = qpe_3ele(prvs[0,1],prvs[1,1],prvs[2,1])
        rainrate[0] = rr1
        rainrate[1] = rr2
        rainrate[2] = rr3
        rainrate[3] = rr4

        '''resnet'''
        logging.info('ResQPE')
        prvs_extent = reshape_3399(prvs)# 3,3,368,1000
        t0 = datetime.datetime.now()
        samples = np.zeros((360*(1000-2*4), 3,3,9,9))# num_samples, 3,3,9,9
        counter = 0
        for true_azi in np.arange(360):
            fake_azi = true_azi+4
            for gate in np.arange(4,1000-4):
                sample = prvs_extent[:,:, fake_azi-4:fake_azi+4+1, gate-4:gate+4+1]
                samples[counter] = sample; counter += 1
        logging.info(f'cost of resample: {datetime.datetime.now()-t0}')
        rr_dl = apply_ResQPE(samples, resnet, device, mode='test')
        rainrate[4, :, 4:-4] = rr_dl.reshape(360, 1000-2*4)

        '''cnnnet'''
        logging.info('CNNQPE')
        prvs_extent = reshape_999(prvs)# 9,368,1000
        t0 = datetime.datetime.now()
        samples = np.zeros((360*(1000-2*4), 9,9,9))# num_samples, 9,9,9
        counter = 0
        for true_azi in np.arange(360):
            fake_azi = true_azi+4
            for gate in np.arange(4,1000-4):
                sample = prvs_extent[:, fake_azi-4:fake_azi+4+1, gate-4:gate+4+1]
                samples[counter] = sample; counter += 1
        logging.info(f'cost of resample: {datetime.datetime.now()-t0}')
        rr_dl = apply_CNNQPE(samples, cnnnet, device)
        rainrate[5, :, 4:-4] = rr_dl.reshape(360, 1000-2*4)

        '''保存数据'''
        np.save(fpsave, rainrate)
        logging.info(fpsave)

        # 在循环结束时输出内存占用情况
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logging.info(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")
        # 输出当前显存占用情况
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        logging.info(f"CUDA memory allocated: {allocated_memory / 1024 ** 2:.2f} MB")
        logging.info(f"CUDA memory reserved: {reserved_memory / 1024 ** 2:.2f} MB")
        # 清理显存
        torch.cuda.empty_cache()
        logging.info('-----------------------------------------')
        # break
    logging.info('Finished!!!!!')


if __name__ == "__main__":
    # torch.backends.cuda.matmul.allow_tf32 = True # 加速：训练测试都行
    # 配置日志记录器
    logging.basicConfig(
        filename='test-run2019.log',                  # 日志文件名
        level=logging.INFO,                   # 记录 INFO 及以上级别的日志
        format='%(asctime)s---%(message)s',   # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S'           # 时间格式
    )
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用的设备:", device)



    '''配置路径'''
    # path_save = './model/based_on_202407/{}'.format('ResQPE-3399-1-vlr-wmse-new_scaler')
    # print(path_save)
    # model = CNNQPE(9,1).to(device)
    # model = ResQPE().to(device)
    # model.load_state_dict(torch.load(path_save + '/' + "model.pth"))#,map_location=torch.device('cpu')))
    
    '''快速测试'''
    # fast_test(path_save, model, device, 'res')

    '''大量测试'''
    # test('/data/zry/radar/Xradar_npy_qpe/BJXSY-test/{}'.format('ResQPE-3399-1-vlr-wmse'), model, device)

    '''run2019'''
    resnet = ResQPE().to(device)
    resnet.load_state_dict(torch.load("model/based_on_202407/ResQPE-3399-1-vlr-wmse-new_scaler-new_stop_3/model.pth"))
    cnnnet = CNNQPE(9,1).to(device)
    cnnnet.load_state_dict(torch.load("model/based_on_202407/CNNQPE-999-1-vlr-wmse-new_scaler-new_stop_3/model.pth"))
    run2019('/data/zry/radar/Xradar_npy_qpe/run2019', resnet, cnnnet, device)
