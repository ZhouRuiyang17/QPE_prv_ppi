import my.utils.mytools as mt
import my.utils.utils_ml as utils
import pandas as pd
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch import nn
import datetime



def mytable(zzz, aaa, bbb, ccc, ddd, eee,
          path_save, thr):
    metrics_hit = pd.DataFrame(columns=['H', 'M', 'F', 'POD', 'FAR', 'CSI'])
    metrics_hit.loc['dl'] = mt.get_metrics_hit(zzz, aaa, thr)
    metrics_hit.loc['ref'] = mt.get_metrics_hit(zzz, bbb, thr)
    metrics_hit.loc['kdp'] = mt.get_metrics_hit(zzz, ccc, thr)
    metrics_hit.loc['refzdr'] = mt.get_metrics_hit(zzz, ddd, thr)
    metrics_hit.loc['kdpzdr'] = mt.get_metrics_hit(zzz, eee, thr)
    metrics_hit.to_csv(f'{path_save}/test-hit-{thr}.csv')


def mask_rr(rr):
    rr[rr<0] = 0
    rr[rr>200] = 0
    rr[np.isnan(rr)] = 0

    return rr

def qpe_3ele(ref, zdr, kdp):
    refup = 10**(ref*0.1)
    zdrup = 10**(zdr*0.1)

    a1 = 0.0576; b1 = 0.557
    a2 = 15.421; b2 = 0.817
    a3 = 0.0059; b3 = 0.994;c3 = -4.929
    # a3 = 0.0061; b3 = 0.959;c3 = -3.671 # zz
    a4 = 26.778; b4 = 0.946;c4 = -1.249
    # a4 = 22.560; b4 = 0.910;c4 = -0.859 # zz



    rr1 = a1*refup**b1
    rr2 = a2*kdp**b2
    rr3 = a3*refup**b3*zdrup**c3
    rr4 = a4*kdp**b4*zdrup**c4

    rr1 = mask_rr(rr1)
    rr2 = mask_rr(rr2)
    rr3 = mask_rr(rr3)
    rr4 = mask_rr(rr4)

    return rr1, rr2, rr3, rr4


def apply_393(data, model, device, center = 4):
    '''筛选'''
    ref = data[:, 3, center-1:center+1+1, center-1:center+1+1]
    refup = 10**(ref*0.1)
    meanup = refup.mean(axis=(1,2))
    mean = 10*np.log10(meanup)
    loc = mean >= 0 
    test_x = data[loc].astype(np.float32)
    # test_x = data.astype(np.float32)

    '''scaler'''
    test_x[:,[0,3,6]] = utils.scaler(test_x[:,[0,3,6]], 'ref').astype(np.float32)
    test_x[:,[1,4,7]] = utils.scaler(test_x[:,[1,4,7]], 'zdr').astype(np.float32)
    test_x[:,[2,5,8]] = utils.scaler(test_x[:,[2,5,8]], 'kdp').astype(np.float32)
    
    '''tensor'''
#     test_x = torch.from_numpy(test_x).to(device)
    test_x = utils.totensor(test_x, device)

#     model = CNN(9,3).to(device)
#     model.load_state_dict(torch.load(path_save + '/' + "cnn.pth"))#,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred = model(test_x)

    pred = utils.toarray(pred, 3)
    pred[:,0] = utils.scaler(pred[:,0], 'rr', 1)
    pred[:,1] = utils.scaler(pred[:,1], 'D0', 1)
    pred[:,2] = utils.scaler(pred[:,2], 'log10Nw', 1)
    # pred = utils.scaler(pred, 'log10rr', 1); pred = 10**(pred)

    rainrate = np.zeros(len(data))
    rainrate[loc] = pred[:,0]
    # rainrate = pred[:,0]
    rainrate = mask_rr(rainrate)
    return rainrate

def apply_CNNQPE(data, model, device, center = 4):
    # data: n,9,9,9
    '''筛选'''
    t0 = datetime.datetime.now()
    ref = data[:, 3, center-1:center+1+1, center-1:center+1+1]
    refup = 10**(ref*0.1)
    meanup = refup.mean(axis=(1,2))
    mean = 10*np.log10(meanup)
    loc = mean >= 0 
    test_x = data[loc].astype(np.float32)
    logging.info(f'cost of location: {datetime.datetime.now()-t0}')
    logging.info(f'num of input: {len(test_x)}')
    
    if len(test_x) != 0:
        '''scaler'''
        test_x[:,[0,3,6]] = utils.scaler(test_x[:,[0,3,6]], 'ref').astype(np.float32)
        test_x[:,[1,4,7]] = utils.scaler(test_x[:,[1,4,7]], 'zdr').astype(np.float32)
        test_x[:,[2,5,8]] = utils.scaler(test_x[:,[2,5,8]], 'kdp').astype(np.float32)
        '''tensor'''
        test_x = utils.totensor(test_x, device)
        '''calculate'''
        t0 = datetime.datetime.now()
        pred = utils.DLcalculate(model, test_x, 1024)
        logging.info(f'cost of QPE: {datetime.datetime.now()-t0}')
        '''tensor-re'''
        pred = utils.toarray(pred, 1)
        '''scaler-re'''
        pred[:,0] = utils.scaler(pred[:,0], 'rr', 1)
        rainrate = np.zeros(len(data))
        rainrate[loc] = pred[:,0]
        rainrate = mask_rr(rainrate)
    else:
        rainrate = np.zeros(len(data))
    return rainrate

def apply_ResQPE(data, model, device, center = 4, mode='fast_test'):
    # data: n,9,9,9
    if mode != 'fast_test':
        datatemp = np.zeros((len(data), 9,9,9))
        datatemp[:, [0,3,6]] = data[:,0,:]
        datatemp[:, [1,4,7]] = data[:,1,:]
        datatemp[:, [2,5,8]] = data[:,2,:]
        data = datatemp
    '''筛选'''
    t0 = datetime.datetime.now()
    ref = data[:, 3, center-1:center+1+1, center-1:center+1+1]
    refup = 10**(ref*0.1)
    meanup = refup.mean(axis=(1,2))
    mean = 10*np.log10(meanup)
    loc = mean >= 0 
    test_x = data[loc].astype(np.float32)
    logging.info(f'cost of location: {datetime.datetime.now()-t0}')
    logging.info(f'num of input: {len(test_x)}')
    
    if len(test_x) != 0:
        '''scaler'''
        test_x[:,[0,3,6]] = utils.scaler(test_x[:,[0,3,6]], 'ref').astype(np.float32)
        test_x[:,[1,4,7]] = utils.scaler(test_x[:,[1,4,7]], 'zdr').astype(np.float32)
        test_x[:,[2,5,8]] = utils.scaler(test_x[:,[2,5,8]], 'kdp').astype(np.float32)
        test_x1 = np.zeros((len(test_x), 3, 3, 9, 9))
        test_x1[:, 0] = test_x[:, [0,3,6]]
        test_x1[:, 1] = test_x[:, [1,4,7]]
        test_x1[:, 2] = test_x[:, [2,5,8]]
        '''tensor'''
        test_x = utils.totensor(test_x1, device)
        '''calculate'''
        t0 = datetime.datetime.now()
        pred = utils.DLcalculate(model, test_x, 1024)
        logging.info(f'cost of QPE: {datetime.datetime.now()-t0}')
        '''tensor-re'''
        pred = utils.toarray(pred, 1)
        '''scaler-re'''
        pred[:,0] = utils.scaler(pred[:,0], 'rr', 1)
        rainrate = np.zeros(len(data))
        rainrate[loc] = pred[:,0]
        rainrate = mask_rr(rainrate)
    else:
        rainrate = np.zeros(len(data))
    return rainrate



def apply_391_resver2(data, model, device, center = 4):
    '''筛选'''
    ref = data[:, 3, center-1:center+1+1, center-1:center+1+1]
    refup = 10**(ref*0.1)
    meanup = refup.mean(axis=(1,2))
    mean = 10*np.log10(meanup)
    loc = mean >= 0 
    test_x = data[loc].astype(np.float32)
    # test_x = data.astype(np.float32)

    '''scaler'''
    test_x[:,[0,3,6]] = utils.scaler(test_x[:,[0,3,6]], 'ref').astype(np.float32)
    test_x[:,[1,4,7]] = utils.scaler(test_x[:,[1,4,7]], 'zdr').astype(np.float32)
    test_x[:,[2,5,8]] = utils.scaler(test_x[:,[2,5,8]], 'kdp').astype(np.float32)
    test_x1 = np.zeros((len(test_x), 3, 3, 9, 9))
    test_x1[:, 0] = test_x[:, [0,3,6]]
    test_x1[:, 1] = test_x[:, [1,4,7]]
    test_x1[:, 2] = test_x[:, [2,5,8]]
    
    '''tensor'''
#     test_x = torch.from_numpy(test_x).to(device)
    test_x = utils.totensor(test_x1, device)

#     model = CNN(9,3).to(device)
#     model.load_state_dict(torch.load(path_save + '/' + "cnn.pth"))#,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred = model(test_x)

    pred = utils.toarray(pred, 1)
    pred[:,0] = utils.scaler(pred[:,0], 'rr', 1)
#     pred[:,1] = utils.scaler(pred[:,1], 'D0', 1)
#     pred[:,2] = utils.scaler(pred[:,2], 'log10Nw', 1)
    # pred = utils.scaler(pred, 'log10rr', 1); pred = 10**(pred)

    rainrate = np.zeros(len(data))
    rainrate[loc] = pred[:,0]
    # rainrate = pred[:,0]
    rainrate = mask_rr(rainrate)
    return rainrate

def apply_resver3(data, model, device, center = 4):
    '''筛选'''
    ref = data[:, 3, center-1:center+1+1, center-1:center+1+1]
    refup = 10**(ref*0.1)
    meanup = refup.mean(axis=(1,2))
    mean = 10*np.log10(meanup)
    loc = mean >= 0 
    test_x = data[loc].astype(np.float32)

    '''scaler'''
    test_x[:,[0,3,6]] = utils.scaler(test_x[:,[0,3,6]], 'ref').astype(np.float32)
    test_x[:,[2,5,8]] = utils.scaler(test_x[:,[2,5,8]], 'kdp').astype(np.float32)
    test_x1 = np.zeros((len(test_x), 2, 3, 9, 9))
    test_x1[:, 0] = test_x[:, [0,3,6]]
    test_x1[:, 1] = test_x[:, [2,5,8]]
    
    '''tensor'''
    test_x = utils.totensor(test_x1, device)

    model.eval()
    with torch.no_grad():
        pred = model(test_x)

    pred = utils.toarray(pred, 1)
    pred[:,0] = utils.scaler(pred[:,0], 'rr', 1)

    rainrate = np.zeros(len(data))
    rainrate[loc] = pred[:,0]
    rainrate = mask_rr(rainrate)
    return rainrate