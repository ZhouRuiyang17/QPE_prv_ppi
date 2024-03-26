import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

from model import CNN
import utils

path = r'E:\QPE_prv_ppi_2_99\dataset\20240326'
path_save = r'E:\QPE_prv_ppi_2_99\model\{}'.format('20240326')
if not os.path.exists(path_save):
    os.makedirs(path_save)


# =============================================================================
# maxi = [70, 7, 7, 1, 100]
# mini = [ 0, 0,  0, 0,   0]
# 
# def scaler(datas):
#     for i, data in enumerate(datas):
#         datas[i] = ml.min_max(data, mini[i], maxi[i])
#     return datas
# 
# class WeightedMSELoss(nn.Module):
#     def __init__(self, weights, edge):
#         super(WeightedMSELoss, self).__init__()
#         self.weights = weights  # 传入的权重
#         self.edge = edge
# 
#     def forward(self, predicted, target):
#         edge = self.edge
#         for i, _ in enumerate(edge[:-1]):
#             loc = np.where((target > edge[i]) & (target <= edge[i+1]))[0]
#             if i == 0:
#                 loss = self.weights[i] * (predicted[loc] - target[loc])**2
#             else:
#                 loss = torch.cat([loss, self.weights[i] * (predicted[loc] - target[loc])**2])
#         # 计算每个样本的损失并加权求和
#         loss = torch.sum(loss) / np.sum(weights)
#         # loss = torch.mean((predicted - target)**2)
#         return loss
# 
# class WeightedMSELoss_ver2(nn.Module):
#     def __init__(self):
#         super(WeightedMSELoss_ver2, self).__init__()
# 
# 
#     def forward(self, predicted, target):
#         loss = []
#         for i in range(len(target)):
#             loss += [target[i]*100 * (predicted[i] - target[i])**2]
#         # 计算每个样本的损失并加权求和
#         loss = torch.Tensor(loss)
#         loss = (torch.mean(loss))
#         loss2 = torch.mean(target*100*(predicted - target)**2)
#         return loss2
# =============================================================================


if __name__ == "__main__":
    
    # ----封装
    train_x = np.load(os.path.join(path,'train_x.npy'), allow_pickle=True).astype(np.float32)
    train_y = np.load(os.path.join(path,'train_y.npy'), allow_pickle=True).reshape(-1, 1).astype(np.float32)
    vali_x = np.load(os.path.join(path,'vali_x.npy'), allow_pickle=True).astype(np.float32)
    vali_y = np.load(os.path.join(path,'vali_y.npy'), allow_pickle=True).reshape(-1, 1).astype(np.float32)
    test_x = np.load(os.path.join(path,'test_x.npy'), allow_pickle=True).astype(np.float32)
    test_y = np.load(os.path.join(path,'test_y.npy'), allow_pickle=True).astype(np.float32)
    
    train = utils.loader(train_x, train_y, 64)
    vali = utils.loader(vali_x, vali_y, 64)
    
    # ----训练
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3, weight_decay = 1e-4)
    loss_func = torch.nn.L1Loss()

    plt.ion()
    plt.show()
    epochs = 500
    loss_train = []; loss_vali = []
    params = []; positive_counter = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        res1, res2 = utils.trainer(train, vali, model, loss_func, optimizer)
        
        loss_train += [res1[-1]]
        loss_vali += [res2[-1]]
        if t % 50 == 0:
            plt.cla()
            rainrate = np.array(res1[1]).flatten(); prediction = np.array(res1[2]).flatten()
            plt.hist2d(rainrate, prediction,bins = [np.arange(0,1,0.01)]*2, norm = colors.LogNorm())
            plt.plot([0,1],[0,1])
            plt.title('training')
            plt.pause(0.5)

            plt.cla()
            rainrate = np.array(res2[1]).flatten(); prediction = np.array(res2[2]).flatten()
            plt.hist2d(rainrate, prediction,bins = [np.arange(0,1,0.01)]*2, norm = colors.LogNorm())
            plt.plot([0,1],[0,1])
            plt.title('training')
            plt.pause(0.5)

            plt.cla()
            plt.plot(loss_train, label = 'train loss')
            plt.plot(loss_vali, label = 'vali loss')
            plt.legend()
            plt.pause(0.5)
            
        if len(params) < epochs/10:
            params.append(model.state_dict())
        else:
            params.append(model.state_dict())
            params = params[1:]
            positive_counter += utils.early_stop(loss_vali, int(epochs/10))
            if positive_counter == 20:
                torch.save(params[-1], path_save + '/' + "cnn.pth")
                print('early stop at epoch:{}'.format(t))
                plt.plot(loss_train, label = 'train loss')
                plt.plot(loss_vali, label = 'vali loss')
                plt.legend()
                break
    print("Done!")
    plt.ioff()
    plt.show()
    
    # [6]
    if positive_counter != 20:
        torch.save(model.state_dict(), path_save + '/' + "cnn.pth")
        print('finish all epochs:{}'.format(epochs))    
    #%%
# =============================================================================
#     # [7]
#     net = CNN()
#     net.load_state_dict(torch.load(path_save + '\\' + "cnn.pth"))
#     net.eval()
#     with torch.no_grad():
#         pred = net(test_x)
#     
#     pred = pred.view(-1).detach().numpy()
#     # pred = np.log(pred + 1)
#     pred = ml.min_max_rev(pred, mini[-1], maxi[-1])
#     # pred = 10**pred
#     
#     # metrics = []
#     # scatter = mytools.Scatter(y_test, zr300)
#     # scatter.plot3(bins = [np.arange(100), np.arange(100)], lim=[[0,100],[0,100]],draw_line = 1)
#     # metrics += [scatter.evaluate().copy()]
#     # df = pd.DataFrame(metrics)
#     
#     metrics_ml = {}
#     # metrics_300 = {}
# 
#     # ----评估
#     scatter = mt.Scatter(test_y, pred)
#     metrics_ml['all'] = scatter.evaluate().copy()
#     scatter.plot3(bins = [np.arange(100), np.arange(100)], lim=[[0.1,100],[0.1,100]],draw_line = 1,
#                   show_metrics=1, label = ['rain rate (gauge) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'ML')
#     
#     # scatter = mytools.Scatter(y_test, zr300)
#     # metrics_300['all'] = scatter.evaluate().copy()
#     # scatter.plot3(bins = [np.arange(100), np.arange(100)], lim=[[1,100],[1,100]],draw_line = 1,
#     #               show_metrics=1, label = ['rain rate (gauge) (mm/h)', 'rain rate (radar) (mm/h)'], title = 'Z-R relation')
# 
#     
#     # ----分段评估
#     # edge = [0.1, 10, 20, 30, 40, 50, 100, 200]
#     # for i in range(len(edge) - 1):     
#     #     loc = np.where((y_test >= edge[i]) & (y_test < edge[i+1]))
#         
#     #     scatter = mytools.Scatter(y_test[loc], pred[loc])
#     #     metrics_ml['{}-{}'.format(str(edge[i]), str(edge[i+1]))] = scatter.evaluate().copy()
# 
#         # scatter = mytools.Scatter(y_test[loc], zr300[loc])
#         # metrics_300['{}-{}'.format(str(edge[i]), str(edge[i+1]))] = scatter.evaluate().copy()
#        
#     # metrics_ml = pd.DataFrame(metrics_ml)
#     # metrics_300 = pd.DataFrame(metrics_300)
#     # metrics = pd.concat([metrics_ml, metrics_300], axis=0)
# 
#     # metrics.to_excel( os.path.join(path_save, 'stat.xlsx'))
#     # metrics_ml.to_excel( os.path.join(path_save, 'statmlp.xlsx'))
#     # metrics_300.to_excel( os.path.join(path_save, 'stat300.xlsx'))
# =============================================================================
