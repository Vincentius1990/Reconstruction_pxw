from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset





class MyDataset(Dataset):
    def __init__(self, data_path, label_path):                       #构造函数
        data_path = data_path
        label_path = label_path
        self.data = np.load(data_path)
        self.label = np.load(label_path)

    def __getitem__(self, index):                                    #用于索引数据集中的数据
        data = self.data[index]
        labels = self.label[index]
#         data = np.expand_dims(data, 0)                           
#         labels = np.expand_dims(labels, 0)                           #将矩阵扩充为4维,样本数，通道数和二维尺寸
        
        data = torch.from_numpy(data)                                #numpy格式转为tensor格式
        data = data.type(torch.FloatTensor)
        labels = torch.tensor(labels)
        labels = labels.type(torch.FloatTensor)
        
#         dataset = TensorDataset(data, labels)
        return data, labels

    def __len__(self):                                              #定义整个数据集的长度
        return len(self.data)


if __name__ == "__main__":

    path = '/mnt/share1/pengxingwen/Dataset/Rec2/'
    train_F = []
    train_u = []
    for idx in range(5000):
        F_data1 = loadmat('/mnt/share1/pengxingwen/rec_1_onepoint/Example'+str(idx)+'.mat')['F']
        u_data1 = loadmat('/mnt/share1/pengxingwen/rec_1_onepoint/Example'+str(idx)+'.mat')['u']
        ft1, ut1 = data_cat(F_data1, u_data1)
        train_F.append(ft1)                   #额定功率热布局和19*19个测点温度
        train_u.append(ut1)                   #真实功率热布局和温度场

    print(train_F[0].shape, train_u[0].shape)  
    print(len(train_F), len(train_u))
    np.save(path + "train_F", train_F)
    np.save(path + "train_u", train_u)

    #----------------------------------------------读取测试集—————————————————————————————————————
    test_F = []
    test_u = []
    for idx in range(1000): 
        F_data2 = loadmat('/mnt/share1/pengxingwen/rec_1_onepoint/Example'+str(5000+idx)+'.mat')['F']
        u_data2 = loadmat('/mnt/share1/pengxingwen/rec_1_onepoint/Example'+str(5000+idx)+'.mat')['u']
        ft2, ut2 = data_cat(F_data2, u_data2)
        test_F.append(ft2)                 #包含热布局和16个测点温度
        test_u.append(ut2)                                     #真实的温度场

    print(test_F[0].shape, test_u[0].shape)
    print(len(test_F),len(test_u))
    np.save(path + "test_F", test_F)
    np.save(path + "test_u", test_u)

    #---------------------------------------------------封装数据集----------------------------------------------------------------

    path = '/mnt/share1/pengxingwen/Dataset/Rec2/'

    #-------------------------------------------构造Dataset对象和DataLoader迭代对象-----------------------------------------------------
    batch_size = 64   #封装的批量大小，一般取64、128或者256
    train_dataset = MyDataset(path + "train_F.npy", path + "train_u.npy")
    train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle= True, num_workers=4)
    test_dataset = MyDataset(path + "test_F.npy", path + "test_u.npy")
    test_iter = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=4)



#     #----------------------------------------------查看数据集效果-------------------------------------------------
#     for i, sample in enumerate(train_iter):
#         if i == 1:
#             break
#         else:
#             F1, u1 = sample
            
#     print('F1', type(F1), F1.shape)
#     print('u1', type(u1), u1.shape)        
#     #---------------------------------------绘制热布局和温度场样本-------------------------------------------------
#     plt.figure(figsize=(15, 15))   #figsize=(40, 40)

#     plt.subplot(4,1,1)                                          #绘制一张空白图
#     plt.imshow(F1[0][0])                                           #显示读入图片
#     plt.axis('off')
#     plt.colorbar()

#     plt.subplot(4,1,2)                                          #绘制一张空白图
#     plt.imshow(F1[0][1])                                           #显示读入图片
#     plt.axis('off')
#     plt.colorbar()

#     plt.subplot(4,1,3)                                           #绘制一张空白图
#     plt.imshow(u1[0][0])
#     plt.axis('off') 
#     plt.colorbar()

#     #-----------------------------------------检验数据能否用于训练-------------------------------------------------------------------------
#     model1 = UNet_VGG()

#     with torch.no_grad():
#         y3 = model1(F1)

#     print('output', y3.shape, type(y3))

#     plt.subplot(4,1,4)                                           #绘制一张空白图
#     plt.imshow(y3[0][0])
#     plt.axis('off') 
#     plt.colorbar()
