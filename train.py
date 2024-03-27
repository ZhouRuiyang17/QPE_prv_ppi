import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import *
import utils

path = './dataset/20240326'
path_save = './model/{}'.format('20240327-15-try3-wmae')
if not os.path.exists(path_save):
    os.makedirs(path_save)
# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备:", device)


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

    plt.cla()
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot()
    rainrate = np.array(res2[1]).flatten(); prediction = np.array(res2[2]).flatten()
    ax.hist2d(rainrate, prediction,bins = [np.arange(0,1,0.01)]*2, norm = colors.LogNorm())
    ax.plot([0,1],[0,1])
    ax.set_title('vali')
    ax.set_aspect('equal')
    plt.savefig(path_save + '/vali_epoch{}.png'.format(t))

    plt.cla()
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot()
    ax.plot(loss_train, label = 'train loss')
    ax.plot(loss_vali, label = 'vali loss')
    ax.set_aspect('auto')
    plt.legend()
    plt.savefig(path_save + '/loss_epoch{}.png'.format(t))

edge = np.array([0,0.1,10,20,30,40,50,100])
weights = np.array([0.1,1,2,4,8,16,32])
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
    train_x = np.load(os.path.join(path,'train_x.npy'), allow_pickle=True).astype(np.float32)[:,1].reshape(-1,1,9,9)
    train_y = np.load(os.path.join(path,'train_y.npy'), allow_pickle=True).reshape(-1, 1).astype(np.float32)
    vali_x = np.load(os.path.join(path,'vali_x.npy'), allow_pickle=True).astype(np.float32)[:,1].reshape(-1,1,9,9)
    vali_y = np.load(os.path.join(path,'vali_y.npy'), allow_pickle=True).reshape(-1, 1).astype(np.float32)
    test_x = np.load(os.path.join(path,'test_x.npy'), allow_pickle=True).astype(np.float32)[:,1].reshape(-1,1,9,9)
    test_y = np.load(os.path.join(path,'test_y.npy'), allow_pickle=True).astype(np.float32)
    
    train = utils.loader(train_x, train_y, device, 64)
    vali = utils.loader(vali_x, vali_y, device, 64)
    test_x = torch.from_numpy(test_x)
    test_y = utils.scaler(test_y, 'rr', 1).reshape(-1)

    
    # ----训练
    model = CNN_tian().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4, weight_decay = 1e-4)
    loss_func = torch.nn.L1Loss()
    loss_func = wmaeloss(weights, edge)

    # plt.ion()
    # plt.show()
    epochs = 500
    loss_train = []; loss_vali = []
    params = []; positive_counter = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        res1, res2 = utils.trainer(train, vali, model, loss_func, optimizer)
        
        loss_train += [res1[-1]]
        loss_vali += [res2[-1]]
        if t % 50 == 0:           
            plot(res1, res2, loss_train, loss_vali)
            
        if len(params) < epochs/10:
            params.append(model.state_dict())
        else:
            params.append(model.state_dict())
            params = params[1:]
            positive_counter += utils.early_stop(loss_vali, int(epochs/10))
            if positive_counter == 20:
                torch.save(params[-1], path_save + '/' + "cnn.pth")
                print('early stop at epoch:{}'.format(t))
                plot(res1, res2, loss_train, loss_vali)
                break
    
    print("Done!")
    # plt.ioff()
    # plt.show()
    
    # [6]
    if positive_counter != 20:
        torch.save(model.state_dict(), path_save + '/' + "cnn.pth")
        print('finish all epochs:{}'.format(epochs))  



    # [7]
    ### model
    model = CNN_tian()
    model.load_state_dict(torch.load(path_save + '/' + "cnn.pth"))
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
    pred = pred.view(-1).detach().numpy()
    pred = utils.scaler(pred, 'rr', 1)
    scatter = utils.Scatter((test_y), (pred))
    scatter.plot3(bins = [np.arange(0,100)]*2, lim=[[0.1,100]]*2,draw_line = 1,
                  show_metrics=1, label = ['rain rate (gauge) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'cnn',
                  fpath = path_save + '/' + 'test-cnn.png')
    scatter = utils.Scatter(np.log10(test_y), np.log10(pred))
    scatter.plot3(bins = [np.arange(-1,2,0.05)]*2, lim=[[-1,2]]*2,draw_line = 1,
                  show_metrics=1, label = ['rain rate (gauge) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'cnn',
                  fpath = path_save + '/' + 'test-cnn-log.png')
    
    ### zr300
    test_x = test_x.numpy()
    ref = utils.scaler(test_x, 'ref', 1)[:,0,4,4]
    ref = 10**(ref*0.1)
    pred_zr = 0.0576 * (ref)**0.557
    scatter = utils.Scatter((test_y), (pred_zr))
    scatter.plot3(bins = [np.arange(0,100)]*2, lim=[[0.1,100]]*2,draw_line = 1,
                  show_metrics=1, label = ['rain rate (gauge) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'zr relation',
                  fpath = path_save + '/' + 'test-zr.png')
    scatter = utils.Scatter(np.log10(test_y), np.log10(pred_zr))
    scatter.plot3(bins = [np.arange(-1,2,0.05)]*2, lim=[[-1,2]]*2,draw_line = 1,
                  show_metrics=1, label = ['rain rate (gauge) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'zr relation',
                  fpath = path_save + '/' + 'test-zr-log.png')

    
