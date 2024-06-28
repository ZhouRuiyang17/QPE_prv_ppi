import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import utils


def lookup(scan):
    _, num_gate = scan[0][0].shape
    left = scan[:, :, -4:, :]
    right = scan[:, :, :4, :]
    scan = np.append(left, scan, axis = 2)
    scan = np.append(scan, right, axis= 2)
    scan[np.isnan(scan)] = -999.
    
    container = np.zeros((6,9,9))*1.
    features = []; azis = []; gates = []
    for azi in range(360):
        pst = azi + 4
        for gate in range(0+4, 1000-4):
            container[0:2] = scan[0,:,pst-4:pst+5, gate-4:gate+5]
            container[2:4] = scan[1,:,pst-4:pst+5, gate-4:gate+5]
            container[4:6] = scan[3,:,pst-4:pst+5, gate-4:gate+5]

            mean1 = 10*np.log10(np.mean(10**(container[0]/10)))
            mean2 = 10*np.log10(np.mean(10**(container[1]/10)))
            if mean1 > 0 or mean2 > 0:
                features += [container.copy()]
                azis += [azi]
                gates += [gate]
    features = np.array(features).astype(np.float32)
    azis = np.array(azis)
    gates = np.array(gates)
    azi_gate = np.vstack([azis, gates]).transpose()
    
    return features, azi_gate

def QPE(features):
    # ----数据归一并tensor化
    features[:,0:2] = utils.scaler(features[:,0:2], 'ref')
    features[:,2:4] = utils.scaler(features[:,2:4], 'zdr')
    features[:,4:6] = utils.scaler(features[:,4:6], 'kdp')
    features = torch.from_numpy(features)

    # ----加载模型并指定设备
    model = CNN()
    model.load_state_dict(torch.load('./model/20240328-9-cnn 6prv-vlr02-check stop when 100 epoch/cnn.pth', map_location=torch.device('cpu')))
    
    # ----计算                      
    model.eval()
    with torch.no_grad():
        pred = model(features)
    
    # ----结果numpy化并逆归一
    pred = pred.view(-1).detach().numpy()
    pred = utils.scaler(pred, 'rr', 1)

    return pred

def main(fpath):
    # fps = fpath.replace('prv','rr')
    # if os.path.exists(fps):
    #     return 1
    
    import datetime
    scan = np.load(fpath)
    t1 = datetime.datetime.now()
    features, azi_gate = lookup(scan)
    t2 = datetime.datetime.now()
    rr = np.zeros((360,1000))*1.
    if features.size > 0:
        pred = QPE(features)
        rr[azi_gate[:,0], azi_gate[:,1]] = pred
        print('CNN-QPE')
    t3 = datetime.datetime.now()
    print(t2-t1, t3-t2)
    
    import my.mytools as mt
    info = mt.BJXSY
    mt.RADAR(rr, 'rr', *info, eles=[1.45]).ppi(0, [75, 15])
    import my.radarsys as rds
    rd = rds.radar(scan[0],scan[1],scan[2],scan[3],scan[4])
    rd.qpe_mayu(1)
    mt.RADAR(rd.rr, 'rr', *info, eles=[1.45]).ppi(0, [75, 15])

    # print(f"QPE: {fps}")
    # np.save(fps, rr)
    
    # return features, azi_gate, pred
    
if __name__ == "__main__":
    

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用的设备:", device)
    
# =============================================================================
#     ls = []
#     for root, dirs, files in os.walk(r'D:\data\beijing\radar\测试radarsys\for_cnn_test-20240328'):
#         for file in files:
#             if '2019' in file and 'prv' in file:
#                 ls += [os.path.join(root, file)]
#                 main(os.path.join(root, file))
# =============================================================================
                
    main(r'D:\data\beijing\radar\测试radarsys\for_cnn_test-20240328\BJXSY.20190526.024133.prv.npy')
