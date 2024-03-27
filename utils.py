import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
'''数据'''
def scaler(data, dtype, reverse = False):
    mins = {'ref':0,
            'zdr':0,
            'kdp':0,
            'rr':0}
    maxs = {'ref':70,
            'zdr':10,
            'kdp':10,
            'rr':100}
    
    if not reverse:
        data_new = (data - mins[dtype]) / (maxs[dtype] - mins[dtype])
        data_new[data_new<0] = 0
        data_new[data_new>1] = 1
    else:
        data_new = (maxs[dtype] - mins[dtype]) * data + mins[dtype]
        
    return data_new

def spliter(x, y, ratio):
    x = x.copy()
    y = y.copy()
    test_size = ratio[-1]/sum(ratio)
    vali_size = ratio[1]/sum(ratio[:-1])
    
    from sklearn.model_selection import train_test_split
    x1     , x_test , y1     , y_test  = train_test_split(x , y,  test_size = test_size)
    x_train, x_vali,  y_train, y_vali  = train_test_split(x1, y1, test_size = vali_size)
    return [x_train, x_vali, x_test, y_train, y_vali, y_test]

'''训练'''
def loader(x, y, device, batch_size = 64):
    x, y = x.copy().astype(np.float32), y.copy().astype(np.float32)
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = 1)
    
    return dataloader

def trainer(train, vali,
            model, loss_function, optimizer):
    
    model.train()
    xxx_train, yyy_train, ppp_train, lll_train = [], [], [], []
    for batch, (x, y) in enumerate(train):
        pred = model(x)
        loss = loss_function(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        xxx_train += x.tolist()
        yyy_train += y.tolist()
        ppp_train += pred.tolist()
        lll_train += [loss.item()]
    lll_train = sum(lll_train)/len(lll_train)
    
    model.eval()
    xxx_vali, yyy_vali, ppp_vali, lll_vali = [], [], [], []
    with torch.no_grad():
        for batch, (x, y) in enumerate(vali):
            pred = model(x)
            loss = loss_function(pred, y)
            
            xxx_vali += x.tolist()
            yyy_vali += y.tolist()
            ppp_vali += pred.tolist()
            lll_vali += [loss.item()]
    lll_vali = sum(lll_vali)/len(lll_vali)
    
    return [xxx_train, yyy_train, ppp_train, lll_train], [xxx_vali, yyy_vali, ppp_vali, lll_vali]

def early_stop(loss_vali, num_check):
    x = np.arange(num_check)
    y = loss_vali[-num_check:]
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    if slope > 0:
        return 1
    else:
        return 0
    
'''other'''
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import gaussian_kde
class Scatter:
    metrics = {}
    fit_x = []
    fit_y = []
    
    def __init__ (self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

        
    def evaluate(self):
        delta = self.y-self.x
        # self.metrics['sum_x'] = np.nansum(self.x)
        # self.metrics['sum_y'] = np.nansum(self.y)
        # self.metrics['avg_x'] = np.nanmean(self.x)
        # self.metrics['avg_y'] = np.nanmean(self.y)
        # self.metrics['std_x'] = np.nanmean(self.x)
        # self.metrics['std_y'] = np.nanmean(self.y)
        
        self.metrics['total num'] = len(delta)
        # self.metrics['bias'] = np.nansum(abs(delta))/np.nansum(self.x)*100
        self.metrics['ME'] = np.nanmean(delta)
        self.metrics['MRE'] = np.nanmean(abs(delta/self.x))
        self.metrics['MBR'] = np.nansum(self.y) / np.nansum(self.x)
        self.metrics['MAE'] = np.nanmean(abs(delta))
        self.metrics['RMSE'] = (np.nanmean(delta**2))**0.5
        self.metrics['STD'] = np.nanstd(delta)
        x = self.x.copy()
        y = self.y.copy()
        # loc = np.where( (np.isnan(x) == False) & (np.isnan(y) == False))
        # x = x[loc]
        # y = y[loc]
        self.metrics['CC'] = np.corrcoef(x, y)[0,1]
        return self.metrics
    
    def fit(self, lim):
        from scipy.optimize import curve_fit
        x = self.x
        y = self.y
        def fun(x, k):
            return k*x
        para, _ = curve_fit(fun, x, y)
        x = np.arange(lim[0],lim[1])
        y = para[0] * x
        return x, y

    def plot3(self, bins = None, label = None, lim = None, title = None, fpath = None,
              show_metrics = False, draw_line = False, equal = 1, mores = None, 
              showsome = False, scatters = None, lines = None, fit=False):
        self.evaluate()
        if showsome:
            import random
            seeds = np.arange(len(self.x))
            seeds = random.sample(seeds.tolist(), int(0.25*len(self.x)))
            xxx = self.x[seeds]
            yyy = self.y[seeds]  
        else:
            xxx = self.x
            yyy = self.y 
        
        fig = plt.figure(figsize=(10,10),dpi=600)
        ax = fig.add_subplot(111)


        
        if bins != None:
            hist2d = ax.hist2d(xxx, yyy, cmap='Spectral_r', norm=colors.LogNorm(), bins=[bins[0], bins[1]])
        else:
            hist2d = ax.hist2d(xxx, yyy, cmap='Spectral_r', norm=colors.LogNorm())
        
        if scatters != None:
            if scatters[-1] == 1:
                for i, _ in enumerate(scatters):
                    ax.scatter(scatters[i][0], scatters[i][1], c = scatters[i][2])
            else:
                ax.scatter(scatters[0], scatters[1], c = scatters[2])
        
        if fit:
            xxx, yyy = self.fit(lim[0])
            ax.plot(xxx, yyy, c='black')
            
        if equal:
            ax.set_aspect('equal')
        cb = fig.colorbar(hist2d[3], ax=ax)
        # print(cb.get_ticks())
        plt.grid()

        if label != None:
            ax.set_xlabel(label[0])
            ax.set_ylabel(label[1])
        if lim != None:
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])
        if title != None:
            plt.title(title)  

        
        if show_metrics:
            ax.text(lim[1][0],0.95*lim[1][1],'NUM = {:.0f}'.format(self.metrics['total num']))
            ax.text(lim[1][0],0.90*lim[1][1],'MBR = {:.2f}'.format(self.metrics['MBR']))
            ax.text(lim[1][0],0.85*lim[1][1],'MAE = {:.2f}'.format(self.metrics['MAE']))
            # ax.text(lim[1][0],0.75*lim[1][1],'RMSE = {:.2f}'.format(self.metrics['RMSE']))
            ax.text(lim[1][0],0.80*lim[1][1],'CORR = {:.2f}'.format(self.metrics['CC']))
            # ax.text(lim[1][0],0.65*lim[1][1],'STD = {:.2f}'.format(self.metrics['STD']))
        if draw_line:
            ax.plot(lim[0], lim[0], c='black')
        if mores != None:
            ax.scatter(mores[0], mores[1])
        
        # ----额外的比较对象
        if lines != None:
            ax.scatter(lines[0], lines[1], c='black')
        

        if fpath != None:
            plt.savefig(fpath)
        plt.show() 
        
    def plot4(self, bins=None, label = None, lim = None, title = None, fpath = None,
              show_metrics = False, draw_line = False, equal = 1, mores = None, 
              showsome = False, scatters = None, lines = None):
        self.evaluate()
        if showsome:
            import random
            seeds = np.arange(len(self.x))
            seeds = random.sample(seeds.tolist(), int(0.25*len(self.x)))
            xxx = self.x[seeds]
            yyy = self.y[seeds]  
        else:
            xxx = self.x
            yyy = self.y 
        
        fig = plt.figure(figsize=(10,10),dpi=600)
        ax = fig.add_subplot(111)
        if scatters != None:
            for i, _ in enumerate(scatters):
                ax.scatter(scatters[i][0], scatters[i][1], c = scatters[i][2])

        
        xy = np.vstack([xxx, yyy])
        kde = gaussian_kde(xy)
        density = kde(xy)
        scatter = plt.scatter(xxx, yyy, c=density)

        if equal:
            ax.set_aspect('equal')
        cb = fig.colorbar(scatter, ax=ax)
        # print(cb.get_ticks())
        plt.grid()
        
        # ax.set_xticks(bins[0])
        # ax.set_yticks(bins[1])
        if label != None:
            ax.set_xlabel(label[0])
            ax.set_ylabel(label[1])
        if lim != None:
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])
        if title != None:
            plt.title(title)  

        
        if show_metrics:
            ax.text(lim[1][0],0.95*lim[1][1],'NUM = {:.0f}'.format(self.metrics['total num']))
            ax.text(lim[1][0],0.90*lim[1][1],'MBR = {:.2f}'.format(self.metrics['MBR']))
            ax.text(lim[1][0],0.85*lim[1][1],'MAE = {:.2f}'.format(self.metrics['MAE']))
            # ax.text(lim[1][0],0.75*lim[1][1],'RMSE = {:.2f}'.format(self.metrics['RMSE']))
            ax.text(lim[1][0],0.80*lim[1][1],'CORR = {:.2f}'.format(self.metrics['CC']))
            # ax.text(lim[1][0],0.65*lim[1][1],'STD = {:.2f}'.format(self.metrics['STD']))
        if draw_line:
            ax.plot(lim[0], lim[0], c='black')
        if mores != None:
            ax.scatter(mores[0], mores[1])
        
        # ----额外的比较对象
        if lines != None:
            ax.scatter(lines[0], lines[1], c='black')
        
        
        if fpath != None:
            plt.savefig(fpath)
        plt.show() 

