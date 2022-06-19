# encoding: utf-8
#    Authors: Xingwen Peng
#    National University of Defense Technology, China
#    Defense Innovation Institute, Chinese Academy of Military Science, China
#    EMAIL: vincent1990@126.com
#    DATE:  Jun 2022
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
#
# Peng, X., Li, X., Gong, Z., Zhao, X., Yao, W., 2022. A Deep Learning Method Based on Partition Modeling For reconstructing Temperature Field. SSRN Journal. https://doi.org/10.2139/ssrn.4065493
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------

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


class MyNewData(Dataset):                                           
    def __init__(self, root, list_path, ind, transform):               
        super().__init__()
        self.root = root                                               
        self.list_path = list_path                                     
        self.ind = ind                                                
        self.sample_files = make_dataset(root, list_path)
        self.transform = transform

    def __getitem__(self, index):                                      
        path = self.sample_files[index]
        monitor_y, ut = self._loader(path, self.ind)
        if self.transform is not None:
            monitor_y, ut = self.transform(monitor_y), self.transform(ut)
            ind = self.transform(self.ind)
        return monitor_y.float(), ut.float()                           

    def __len__(self):                                                  
        return len(self.sample_files)

    def _loader(self, path, ind):                                     
        ut = loadmat(path)['u']
        monitor_y = ut * ind
        ut = np.expand_dims(ut, axis=0)
        return torch.tensor(monitor_y), torch.tensor(ut)
        # return torch.tensor((monitor_y - 298) / 50), torch.tensor((ut - 298) / 50)    # 归一化
       

if __name__ == '__main__':
    root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'
    train_path = '/mnt/share1/pengxingwen/Dataset/vp/train.txt'
    test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'

    batch_size = 64                                    
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
