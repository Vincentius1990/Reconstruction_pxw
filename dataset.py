import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as mp
from scipy.io import loadmat 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split


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


class SVFRData(Dataset):                                              # 继承Dataset
    '''
    传感器位置可变，双通道输入
    '''

    def __init__(self, root, list_path, ind, transform):               # __init__是初始化该类的一些基础参数
        super().__init__()
        self.root = root                                               # 文件目录
        self.list_path = list_path                                     # 文件list
        self.ind = ind                                                 # 测点提取遮罩
        self.sample_files = make_dataset(root, list_path)
        self.transform = transform

    def __getitem__(self, index):                                      # 根据索引index返回dataset[index]
        path = self.sample_files[index]
        monitor_y, ut = self._loader(path, self.ind)
        if self.transform is not None:
            monitor_y, ut = self.transform(monitor_y), self.transform(ut)
            ind = self.transform(self.ind)
        return monitor_y.float(), ut.float()                            # 返回测点和温度场

    def __len__(self):                                                  # 返回整个数据集的大小
        return len(self.sample_files)

    def _loader(self, path, ind):                                       # 提取测点
        ut = loadmat(path)['u']
        monitor_y = ut * ind
        obs = monitor_y[monitor_y>0].reshape(4,4)
        mask = np.zeros((200, 200))
        for i in range(4):
            for j in range(4):
                mask[50 * i: 50 * i + 50, 50 * j : 50 * j + 50] = obs[i,j]
        mask = np.expand_dims(mask, axis=0)
        ob_mask = np.concatenate((mask, ind), axis = 0)
        ut = np.expand_dims(ut, axis=0)       
        return torch.tensor(ob_mask), torch.tensor(ut)
        # return torch.tensor((monitor_y - 298) / 50), torch.tensor((ut - 298) / 50)    # 归一化


class HeatsinkData(Dataset):                                              # 继承Dataset
    '''
    新的封装类，用列表读取样本
    '''

    def __init__(self, root, list_path, ind, transform):               # __init__是初始化该类的一些基础参数
        super().__init__()
        self.root = root                                               # 文件目录
        self.list_path = list_path                                     # 文件list
        self.ind = ind                                                 # 测点提取遮罩
        self.sample_files = make_dataset(root, list_path)
        self.transform = transform

    def __getitem__(self, index):                                      # 根据索引index返回dataset[index]
        path = self.sample_files[index]
        monitor_y, ut = self._loader(path, self.ind)
        if self.transform is not None:
            monitor_y, ut = self.transform(monitor_y), self.transform(ut)
            ind = self.transform(self.ind)
        return monitor_y.float(), ut.float()                            # 返回测点和温度场

    def __len__(self):                                                  # 返回整个数据集的大小
        return len(self.sample_files)

    def _loader(self, path, ind):                                       # 提取测点
        ut = loadmat(path)['u']
        monitor_y = ut * ind
        monitor_y = (monitor_y[monitor_y != 0] - 298) / 50              # 提取温度测点数据
        u_area = loadmat(path)['u'][0:2, 86:112]                        # 提取热沉区域26*2温度数据
        u_area = (u_area.reshape(52) - 298) / 50
        # ut = np.expand_dims(ut, axis=0)
        return torch.tensor(monitor_y), torch.tensor(u_area)


class MyNewData(Dataset):                                              #继承Dataset
    '''
    新的封装类，用列表读取样本
    '''

    def __init__(self, root, list_path, ind, transform):               # __init__是初始化该类的一些基础参数
        super().__init__()
        self.root = root                                               # 文件目录
        self.list_path = list_path                                     # 文件list
        self.ind = ind                                                 # 测点提取遮罩
        self.sample_files = make_dataset(root, list_path)
        self.transform = transform

    def __getitem__(self, index):                                      # 根据索引index返回dataset[index]
        path = self.sample_files[index]
        monitor_y, ut = self._loader(path, self.ind)
        if self.transform is not None:
            monitor_y, ut = self.transform(monitor_y), self.transform(ut)
            ind = self.transform(self.ind)
        return monitor_y.float(), ut.float()                            # 返回测点和温度场

    def __len__(self):                                                  # 返回整个数据集的大小
        return len(self.sample_files)

    def _loader(self, path, ind):                                       # 提取测点
        ut = loadmat(path)['u']
        monitor_y = ut * ind
        ut = np.expand_dims(ut, axis=0)
        return torch.tensor(monitor_y), torch.tensor(ut)
        # return torch.tensor((monitor_y - 298) / 50), torch.tensor((ut - 298) / 50)    # 归一化
       

if __name__ == '__main__':
    root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'
    train_path = '/mnt/share1/pengxingwen/Dataset/vp/train.txt'
    test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'

    batch_size = 1                                    # 封装的批量大小，一般取16、32、64、128或者256
    ind_4 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_4.pt')
    # train_dataset = HeatsinkData(root, train_path, ind_4, None)
    # train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle= True, num_workers=4)
    test_dataset = HeatsinkData(root, test_path, ind_4, None)
    test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=4)
    # print(len(train_iter), len(test_iter))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, batch_data in enumerate(test_iter):
        # print(i)                                          
        X1, y1 = batch_data
        print('X1:', X1.shape, 'y1:', y1.shape)
        y1 = y1.numpy()
        y1 = y1.reshape(5, 30)
        plt.imshow(y1, cmap='jet')  
        plt.colorbar(fraction=0.045, pad=0.05)
        plt.savefig('y1_' + str(i) + '.png')   
        if i == 1:
            break                                    
