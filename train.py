import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import utils

path = './dataset/20240509'
path_save = './model/based_on_20240509/{}'.format('240509-cnn 3prv-02per10')
if not os.path.exists(path_save):
    os.makedirs(path_save)
# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备:", device)

input(f'res will be stored in:\n{path_save}\nshall we go on[y/n]?')

def plot(res1, res2, loss_train, loss_vali):
    t = len(loss_train)

    plt.cla()
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot()
    rainrate = np.array(res1[1]).flatten(); prediction = np.array(res1[2]).flatten()
    ax.hist2d(rainrate, prediction,bins = [np.arange(0,1,0.01)]*2, norm = colors.LogNorm())
    ax.plot([0,1],[0,1])
    ax.set_title('train')
    ax.set_aspect('equal')
    plt.savefig(path_save + '/train_epoch{}.png'.format(t))
    plt.close()

    plt.cla()
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot()
    rainrate = np.array(res2[1]).flatten(); prediction = np.array(res2[2]).flatten()
    ax.hist2d(rainrate, prediction,bins = [np.arange(0,1,0.01)]*2, norm = colors.LogNorm())
    ax.plot([0,1],[0,1])
    ax.set_title('vali')
    ax.set_aspect('equal')
    plt.savefig(path_save + '/vali_epoch{}.png'.format(t))
    plt.close()

    plt.cla()
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot()
    ax.plot(loss_train, label = 'train loss')
    ax.plot(loss_vali, label = 'vali loss')
    ax.set_aspect('auto')
    plt.legend()
    plt.savefig(path_save + '/loss.png')
    plt.close()

def qpe_mayu(ref, zdr, kdp):
    a1 = 0.0576; b1 = 0.557
    a2 = 15.421; b2 = 0.817
    a3 = 0.0059; b3 = 0.994;c3 = -4.929
    a4 = 26.778; b4 = 0.946;c4 = -1.249

    
    refup = 10**(ref/10)
    zdrup = 10**(zdr/10)
    rr = np.zeros(ref.shape)
    
    loc1 = np.where((kdp <= 0.3) & (zdr <= 0.25))
    loc3 = np.where((kdp <= 0.3) & (zdr > 0.25))
    loc2 = np.where((kdp > 0.3) & (zdr <= 0.25))
    loc4 = np.where((kdp > 0.3) & (zdr > 0.25))

    rr[loc1] = a1*refup[loc1]**b1
    rr[loc3] = a3*refup[loc3]**b3*zdrup[loc3]**c3
    rr[loc2] = a2*kdp[loc2]**b2
    rr[loc4] = a4*kdp[loc4]**b4*zdrup[loc4]**c4
    
    return rr

edge = np.array([0,10,20,30,40,50,100])
weights = np.array([1,2,3,4,5,10])
class wmaeloss(nn.Module):  
    def __init__(self, weights, edge):  
        super(wmaeloss, self).__init__()  
        self.weights = weights
        self.edge = edge
  
    def forward(self, y, p):  
        '''
        y and p is deviced
        diff and loss_list need to be deviced
        '''
        diff = torch.abs(p-y).to(device)
        loss_list = torch.tensor([]).to(device)
        for i in range(len(self.weights)):
            loc = (y >= self.edge[i]) & (y < self.edge[i+1])
            # print(loc)
            loss_list = torch.cat([loss_list, self.weights[i] * diff[loc]])
            # print(loss_list)
        loss = torch.sum(loss_list) / len(loss_list)

        # return diff, loss_list, loss
        return loss

if __name__ == "__main__":
    
    # ----封装
    train_x = np.load(os.path.join(path,'train_x.npy'), allow_pickle=True).astype(np.float32)#[:,1].reshape(-1,1,9,9)
    train_y = np.load(os.path.join(path,'train_y.npy'), allow_pickle=True).reshape(-1, 1).astype(np.float32)
    vali_x = np.load(os.path.join(path,'vali_x.npy'), allow_pickle=True).astype(np.float32)#[:,1].reshape(-1,1,9,9)
    vali_y = np.load(os.path.join(path,'vali_y.npy'), allow_pickle=True).reshape(-1, 1).astype(np.float32)
    test_x = np.load(os.path.join(path,'test_x.npy'), allow_pickle=True).astype(np.float32)#[:,1].reshape(-1,1,9,9)
    test_y = np.load(os.path.join(path,'test_y.npy'), allow_pickle=True).astype(np.float32)

    # train_x = train_x[:, 1::2]
    # vali_x = vali_x[:, 1::2]
    # test_x = test_x[:, 1::2]
    
    train = utils.loader(train_x, train_y, device, 64)
    vali = utils.loader(vali_x, vali_y, device, 64)
    test_x = torch.from_numpy(test_x)
    test_y = utils.scaler(test_y, 'rr', 1).reshape(-1)

    
    # ----训练
    model = CNN_3prv().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4, weight_decay = 1e-4)
    loss_func = torch.nn.L1Loss()
    loss_func = wmaeloss(weights, edge)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)


    epochs = 500
    loss_train = []; loss_vali = []
    params = []; slopes = []; positive_counter = 0; positive_position = []
    for t in range(epochs):
        print(f"-------------------------------\nEpoch {t+1}")
        
        res1, res2 = utils.trainer(train, vali, model, loss_func, optimizer)
        scheduler.step()
        print(f"Epoch {t + 1}, Learning Rate:")
        for param_group in optimizer.param_groups:
            print(param_group['lr']) # 打印更新后的学习率
        
        loss_train += [res1[-1]]
        loss_vali += [res2[-1]]
        if t % 50 == 0 or t == 5:           
            plot(res1, res2, loss_train, loss_vali)
        
        '''
        always store the LAST epochs/10 of params
        check the slope of the loss_vali of the LAST epochs/10 
        when slope > 0, count, and record the SLOPE and POSITION
        when counter == 20, stop
        '''
        if len(params) < epochs/10:
            params.append(model.state_dict())
        else:
            params.append(model.state_dict())
            params = params[1:]
            flag, slope = utils.early_stop(loss_vali, int(epochs/10))
            if flag and t >= 100:
                slopes += [slope]
                positive_position += [t]
                positive_counter += flag
            if positive_counter == 20:
                torch.save(params[-1], path_save + '/' + "cnn.pth")
                print('early stop at epoch:{}'.format(t))
                plot(res1, res2, loss_train, loss_vali)
                print(slopes)
                print(positive_position)
                break
            # flag_stop = utils.early_stop_ptrend(loss_vali, 10)
            # if flag_stop:
            #     torch.save(params[-1], path_save + '/' + "cnn.pth")
            #     print('early stop at epoch:{}'.format(t))
            #     plot(res1, res2, loss_train, loss_vali)
            #     break
    
    print("Done!")

    
    
    if positive_counter != 20:
        torch.save(model.state_dict(), path_save + '/' + "cnn.pth")
        print('finish all epochs:{}'.format(epochs))  



    
    ### model
    model = CNN_3prv()
    model.load_state_dict(torch.load(path_save + '/' + "cnn.pth"))#,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
    pred = pred.view(-1).detach().numpy()
    pred = utils.scaler(pred, 'rr', 1)
    scatter = utils.Scatter((test_y), (pred))
    scatter.plot3(bins = [np.arange(0,100)]*2, lim=[[0.1,100]]*2,draw_line = 1,
                  show_metrics=True, label = ['rain rate (ground) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'cnn',
                  fpath = path_save + '/' + 'test-cnn 20240507.png')
    # scatter = utils.Scatter(np.log10(test_y), np.log10(pred))
    # scatter.plot3(bins = [np.arange(-1,2,0.05)]*2, lim=[[-1,2]]*2,draw_line = 1,
    #               show_metrics=1, label = ['rain rate (ground) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'cnn',
    #               fpath = path_save + '/' + 'test-cnn-log.png')
    



    ### zr300
    test_x = test_x.numpy()
    # ref = utils.scaler(test_x[:,1], 'ref', 1)[:,4,4]
    # zdr = utils.scaler(test_x[:,3], 'zdr', 1)[:,4,4]
    # kdp = utils.scaler(test_x[:,5], 'kdp', 1)[:,4,4]
    ref = utils.scaler(test_x[:,0], 'ref', 1)[:,4,4]
    zdr = utils.scaler(test_x[:,1], 'zdr', 1)[:,4,4]
    kdp = utils.scaler(test_x[:,2], 'kdp', 1)[:,4,4]
    pred_prv = qpe_mayu(ref, zdr, kdp)
    scatter = utils.Scatter((test_y), (pred_prv))
    scatter.plot3(bins = [np.arange(0,100)]*2, lim=[[0.1,100]]*2,draw_line = 1,
                  show_metrics=True, label = ['rain rate (ground) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'prv',
                  fpath = path_save + '/' + 'test-prv 20240507.png')
    # scatter = utils.Scatter(np.log10(test_y), np.log10(pred_prv))
    # scatter.plot3(bins = [np.arange(-1,2,0.05)]*2, lim=[[-1,2]]*2,draw_line = 1,
    #               show_metrics=1, label = ['rain rate (ground) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'prv',
    #               fpath = path_save + '/' + 'test-prv-log.png')

    
    # plt.figure(figsize=(3,3),dpi=600)
    # plt.boxplot([pred_prv-test_y, pred-test_y], labels=['prv', 'cnn'], showfliers=0,showmeans=1)
    # plt.grid()
    # plt.title('BIAS')
    # plt.show()
    # plt.close()

    # rmaes_model = {}
    # edge = [0,1,10,20,30,40,50,100]
    # for i in range(len(edge)-1):
    #     loc = np.where((test_y >= edge[i]) & (test_y < edge[i+1]))[0]
    #     if len(loc) > 0:
    #         key = '{} - {}'.format(edge[i], edge[i+1])
    #         rmaes_model[key] = (utils.Scatter((test_y[loc]), (pred[loc])).evaluate())['RMAE']
    # rmaes_prv = {}
    # edge = [0,1,10,20,30,40,50,100]
    # for i in range(len(edge)-1):
    #     loc = np.where((test_y >= edge[i]) & (test_y < edge[i+1]))[0]
    #     if len(loc) > 0:
    #         key = '{} - {}'.format(edge[i], edge[i+1])
    #         rmaes_prv[key] = (utils.Scatter((test_y[loc]), (pred_prv[loc])).evaluate())['RMAE']
            
    # rmbs_model = {}
    # edge = [0,1,10,20,30,40,50,100]
    # for i in range(len(edge)-1):
    #     loc = np.where((test_y >= edge[i]) & (test_y < edge[i+1]))[0]
    #     if len(loc) > 0:
    #         key = '{} - {}'.format(edge[i], edge[i+1])
    #         rmbs_model[key] = (utils.Scatter((test_y[loc]), (pred[loc])).evaluate())['RMB']
    # rmbs_prv = {}
    # edge = [0,1,10,20,30,40,50,100]
    # for i in range(len(edge)-1):
    #     loc = np.where((test_y >= edge[i]) & (test_y < edge[i+1]))[0]
    #     if len(loc) > 0:
    #         key = '{} - {}'.format(edge[i], edge[i+1])
    #         rmbs_prv[key] = (utils.Scatter((test_y[loc]), (pred_prv[loc])).evaluate())['RMB']
            
    # plt.rcParams.update({'font.size': 12})
    # fig, ax = plt.subplots()
    # ax.plot(rmbs_prv.values(),c='blue')
    # ax.plot(rmbs_model.values(),c='red')
    # ax2 = ax.twinx()
    # ax2.plot(rmaes_prv.values(),c='blue',linestyle='--')
    # ax2.plot(rmaes_model.values(),c='red',linestyle='--')
    # ax.set_ylabel('RMB (solid)')
    # ax2.set_ylabel('RMAE (dashed)')
    # plt.xticks(range(len(edge)-1),rmaes_model.keys())
    # ax.set_xlabel('Rain rate inteval (mm/h)')
    # plt.grid()