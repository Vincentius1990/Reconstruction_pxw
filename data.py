import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as mp
from scipy.io import loadmat 
from torch.utils.data import Dataset, DataLoader, random_split


def sample_plot(dataset_name, obs_name, n, ind, cnd):
    '''
    读取单个mat文件并查看结果
    '''

    data = loadmat("/mnt/share1/pengxingwen/Dataset/" + str(dataset_name[0:2]) + '/' + str(dataset_name) + "/Example" + str(n) + ".mat")
    # print(data.keys() ,data.values())              #输出数据集关键字
    u, Fc = data["u"], data["F"]                     #温度场、监测点和载荷

    ind = ind.reshape(200,200)                       #遮罩降维
    monitor_y = u * ind
    print('Fc', Fc.shape, type(Fc))
    print('u',  u.shape, type(u))
    print('monitor_y',  monitor_y.shape, type(monitor_y))

    # plt_item = [Fc, monitor_y, u]      #真实功率布局,监测点温度,组件温度,真实温度场
    # plt_title = ['Power Intensities (W/m$^2$)', 'Observation Points(K)', 'Real Temperature Field(K)']
    # _a, _b = 23, 5   # 3图 17, 5

    plt_item = [Fc]                         #真实功率布局
    plt_title = ['Power Intensities (W/m$^2$)']
    _a, _b = 6, 5.4   # 1图

    # plt_item = [Fc, u]                    #真实功率布局, 真实温度场
    # plt_title = ['Power Intensities (W/m$^2$)', 'Real Temperature Field(K)']
    # _a, _b = 16, 6.1   # 2图
    
    fig_plot(_a, _b, plt_title, plt_item, dataset_name, obs_name, n)


def cbcolor(p):
    '''
    颜色条设置
    '''
    cb = plt.colorbar(fraction=0.045, pad=0.05)
    cb.ax.tick_params(labelsize=13, rotation=0)
    med = np.max(p) - np.min(p)
    cb.set_ticks([np.min(p), np.min(p) + 0.2 * med, np.min(p) + 0.4 * med, np.min(p) + 0.6 * med, np.min(p) + 0.8 * med, np.max(p)])  #np.max(p)
    cb.update_ticks()


def fig_plot(a, b, plt_title, plt_item, dataset_name, obs_name, n):
    '''
    绘图显示结果
    '''

    fig = plt.figure(figsize=(a, b))  # 绘制样本采用图幅尺寸    
    # fig = plt.figure(figsize=(6.8*len(plt_item), 5.4))                #设置幅面大小,a为图形的宽,b为图形的高,单位为英寸
    # fig.suptitle(str(dataset_name)+'_'+str(obs_name)+' sample')       #图片标题
    xa = np.linspace(0, 0.1, 200)
    ya = np.linspace(0, 0.1, 200)  
    for i, p in enumerate(plt_item):
        plt.subplot(1, len(plt_item), i + 1)                            #绘制子图
        plt.title(plt_title[i], fontsize=20)

        if i == 0:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='summer')            # cmap='jet' 热力图, cmap='nipy_spectral'云图
            cbcolor(p)
            # plt.scatter(x_ob, y_ob, s=20, color='cyan', marker='s')     # 散点,表示测点
            # plt.imshow(p, cmap='gray_r')                                # 无法调换坐标
        elif i == 1:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet')        # cmap='jet' 
            cbcolor(p)
            # plt.scatter(x_ob, y_ob, s=20, color='cyan', marker='s')
        else:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet')        # cmap='jet' 
            cbcolor(p)
        plt.axis('on')
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        # frame = plt.gca()
        # y 轴可见
        # frame.axes.get_yaxis().set_visible(False)
        # x 轴可见
        # frame.axes.get_xaxis().set_visible(False)

    if not os.path.exists('sample'):
        os.makedirs('sample') 
    plt.savefig("./sample/sample_" + str(dataset_name) + "_" + str(obs_name) + "-" + str(n) + ".png",bbox_inches='tight', pad_inches=0.3)
   

class MyNewData(Dataset):                                              #继承Dataset
    def __init__(self, root, list_path, ind, transform):               #__init__是初始化该类的一些基础参数
        super().__init__()
        self.root = root                                               #文件目录
        self.list_path = list_path                                     #文件list
        self.ind = ind                                                 #测点提取遮罩
        self.sample_files = make_dataset(root, list_path)
        self.transform = transform

    def __getitem__(self, index):                                      #根据索引index返回dataset[index]
        path = self.sample_files[index]
        monitor_y, ut = self._loader(path, self.ind)
        if self.transform is not None:
            monitor_y, ut = self.transform(monitor_y), self.transform(ut)
            ind = self.transform(self.ind)
        return monitor_y.float(), ut.float()                            #返回测点和温度场

    def __len__(self):                                                  #返回整个数据集的大小
        return len(self.sample_files)

    def _loader(self, path, ind):                                       #提取测点
        ut = loadmat(path)['u']
        monitor_y = ut * ind
        ut = np.expand_dims(ut, axis=0) 
        return torch.tensor(monitor_y), torch.tensor(ut)
       
def make_dataset(root_dir, list_path):
    files = []
    root_dir = os.path.expanduser(root_dir)

    assert os.path.isdir(root_dir), root_dir
    with open(list_path, 'r') as rf:
        for line in rf.readlines():
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            files.append(path)
        # print(files)
    return files


#———————————————————————————————构造Dataset对象和DataLoader迭代对象—————————————————————————
def dataset_test(root, train_path, test_path, ind):
    '''
    训练集、测试集检验
    '''

    batch_size = 1                                    #封装的批量大小，一般取16、32、64、128或者256    
    train_dataset = MyNewData(root, train_path, ind, None)
    train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle= True, num_workers=4)
    test_dataset = MyNewData(root, test_path, ind, None)
    test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=4)
    print('train_dataset:', len(train_iter), 'test_dataset:',len(test_iter))

    for i, batch_data in enumerate(test_iter):                                        
        X1, y1 = batch_data
        # print('X1:', X1.shape, 'y1:', y1.shape)
        X1, y1 = X1.numpy(), y1.numpy()
        X1, y1 = X1.reshape(200,200), y1.reshape(200,200)
        
        plt_item = [X1, y1]              #真实功率布局,监测点温度,组件温度,真实温度场
        plt_title = ['Observation Points (K)', 'Real Temperature Field (K)']
        fig_plot(plt_title, plt_item, 'vp_c1_60k_test', '4*4ob', i) 
        if i == 1:
            break                                    


root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'
train_path = '/mnt/share1/pengxingwen/Dataset/vp/train.txt'
test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'
ind = torch.load('./src/data/ind_4.pt')

# for i in range(20):
#     sample_plot('vp_c3_55k', '4ob', 16000+i, ind, None)                 #绘制样本

# sample_plot('vp_c1_60k', '4ob', 16000, ind, None)                 #绘制样本
# sample_plot('vp_c3_55k', '4ob', 16003, ind, None)                 #绘制样本

sample_plot('vp_c1_sp', '4ob', 1024, ind, None)                 #绘制样本
sample_plot('vp_c1_sp', '4ob', 20, ind, None)                 #绘制样本
sample_plot('vp_c1_sp', '4ob', 1025, ind, None)                 #绘制样本
sample_plot('vp_c3_sp', '4ob', 1026, ind, None)                 #绘制样本
sample_plot('vp_c3_sp', '4ob', 12, ind, None)                 #绘制样本
sample_plot('vp_c3_sp', '4ob', 1027, ind, None)                 #绘制样本

# dataset_test(root, train_path, test_path, ind_4)                  #训练集、测试集检验
