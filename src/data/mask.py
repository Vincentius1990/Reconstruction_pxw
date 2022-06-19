import torch
import matplotlib.pyplot as plt
import matplotlib.pylab as mp
from scipy.io import loadmat 
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import random
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_check(n):
    '''
    查看单个mat文件样本
    '''

    data = loadmat('/mnt/share1/pengxingwen/Dataset/vp/vp_10_50000/Example' + str(n) + '.mat')
    print(data.keys(), '\n', data.values())                  #输出数据集关键字
    u, Fc = data["u"], data["F"]    #温度场、监测点和载荷
    print('Fc', Fc.shape, type(Fc))
    print('u', u.shape, type(u))


def component_loss_check():
    '''
    测试组件温度MAE loss
    '''

    criterion = torch.nn.L1Loss()
    loss_c = loss.ComponentLoss() 

    cnd = torch.load('./cnd.pt')  
    cnd = cnd.to(device)
    X1= torch.randn(16, 1, 200, 200)
    y1= torch.randn(16, 2, 200, 200)
    X1 = X1.to(device)
    y1 = y1.to(device) 
    y_F1, y_u1 = y1.chunk(2, dim=1)                     #真实功率布局， 组件温度， 真实温度场 
    cnd = cnd.expand(y_u1.size(0), 1, 200, 200) 
    net = FPN_ResNet18(in_channels=1) 
    net = net.to(device)   
    output1 = net(X1)
                
    lt1 = criterion(y_u1, output1)
    lc1 = loss_c(y_u1, output1, cnd)      
    l1 = lt1 + 0.2 * lc1
    print('lt1: {:.4f}  lc1: {:.4f}  l1: {:.4f}'.format(lt1.item(), lc1.item(), l1.item()))

    
def data_cat(Fc, u, ind, cnd):
    '''
    合并监测点、组件温度、热布局与真实温度场
    '''

    Fc = np.expand_dims(Fc, 0)
    u = np.expand_dims(u, 0)
    monitor_y = u * ind
    if cnd is not None:
        ct = u * cnd
        ut = np.concatenate((Fc, ct, u), axis = 0)
    ut = np.concatenate((Fc, u), axis = 0)
    return monitor_y, ut


def fig_plot(plt_title, plt_item, dataset_name, obs_name, n):
    '''
    绘图显示结果
    '''

    fig = plt.figure(figsize=(20, 5.4))                                                #设置幅面大小,a为图形的宽,b为图形的高,单位为英寸
    # fig.suptitle(str(dataset_name) + '_' + str(obs_name) + ' sample')                 #图片标题
    xa = np.linspace(0, 0.1, 200)
    ya = np.linspace(0.1, 0, 200)  
    for i, p in enumerate(plt_item):
        plt.subplot(1, len(plt_item), i + 1)                                          #绘制子图
        plt.title(plt_title[i])
        # plt.imshow(p, cmap='nipy_spectral')                                           #颜色风格，cmap='rainbow', 'jet', 'nipy_spectral'
        plt.contourf(xa, ya, p, levels=150, cmap='jet')   #gray 
        plt.axis('off')
        cb = plt.colorbar(fraction=0.045, pad=0.05)                                   
        v_min = np.min(p)          
        med = np.max(p) - v_min
        cb.set_ticks([v_min, v_min + 0.2 * med, v_min + 0.4 * med, v_min + 0.6 * med, v_min + 0.8 * med, np.max(p)])
        cb.update_ticks()
    plt.savefig("sample_" + str(dataset_name) + "_" + str(obs_name) + "-" + str(n) + ".png",bbox_inches='tight', pad_inches=0.3)


def sample_plot(dataset_name, obs_name, n, ind, cnd):
    '''
    读取单个mat文件并查看结果
    '''

    data = loadmat("/mnt/share1/pengxingwen/Dataset/" + str(dataset_name[0:2]) + '/' + str(dataset_name) + "/Example" + str(n) + ".mat")
    # print(data.keys() ,data.values())                   #输出数据集关键字
    u, Fc = data["u"], data["F"]                          #温度场、监测点和载荷
    print('The ' + str(n) + 'th Raw Mat Information:')
    print('Fc', Fc.shape, type(Fc))
    print('u', u.shape, type(u))

    ft0, ut0 = data_cat(Fc, u, ind, cnd)
    print('The ' + str(n) + 'th Sample Information:')
    print('ft', ft0.shape, type(ft0))
    print('ut', ut0.shape, type(ut0))
    plt_item = [ut0[0], ft0[0], ut0[1]]   #真实功率布局,监测点温度,组件温度,真实温度场
    plt_title = ['Power (W)', 'Observation Points (K)', 'Real Temperature Field (K)']
    fig_plot(plt_title, plt_item, dataset_name, obs_name, n)


def make_ind(n):
    '''
    制作监测点遮罩, 默认提取n*n个监测点
    '''

    ind = np.zeros((400, 400))
    
    if n == 20:
        for i in range(n):
            for j in range(n):
                ind[4 + i * 10][4 + j * 10] = 1 
    
    if n == 19:
        for i in range(n):
            for j in range(n):
                ind[9 + i * 10][9 + j * 10] = 1 

    if n == 15:
        for i in range(n):
            for j in range(n):
                ind[8 + i * 13][8 + j * 13] = 1 


    if n == 16:     #组件中心点 + 边界
        ind[38][182], ind[184][157], ind[89][28], ind[159][49], ind[142][176] = 1, 1, 1, 1, 1
        ind[70][65], ind[41][130], ind[92][157], ind[119][109], ind[42][27] = 1, 1, 1, 1, 1
        ind[0][66], ind[0][133], ind[99][0], ind[99][199], ind[199][66], ind[199][133] = 1, 1, 1, 1, 1, 1

        # ind[50][80], ind[56][156], ind[86][166], ind[80][50], ind[120][100] = 1, 1, 1, 1, 1
        # ind[70][130], ind[120][60], ind[90][94], ind[126][140], ind[66][106] = 1, 1, 1, 1, 1


    if n == 10:     #组件中心点
        # ind[38][182], ind[184][157], ind[89][28], ind[159][49], ind[142][176] = 1, 1, 1, 1, 1
        # ind[70][65], ind[41][130], ind[92][157], ind[119][109], ind[42][27] = 1, 1, 1, 1, 1

        ind[50][80], ind[56][156], ind[86][166], ind[80][50], ind[120][100] = 1, 1, 1, 1, 1
        ind[70][130], ind[120][60], ind[90][94], ind[126][140], ind[66][106] = 1, 1, 1, 1, 1

    # if n == 10:
    #     for i in range(n):
    #         for j in range(n):
    #             ind[9 + i * 20][9 + j * 20] = 1 

    # if n == 8:
    #     for i in range(n):
    #         for j in range(n):
    #             ind[13 + i * 25][13 + j * 25] = 1

    if n == 8:
        for i in range(n):
            for j in range(n):
                ind[24 + i * 50][24 + j * 50] = 1

    if n == 6:
        for i in range(n):
            for j in range(n):
                ind[16 + i * 33][16 + j * 33] = 1 

    if n == 5:
        for i in range(n):
            for j in range(n):
                ind[19 + i * 40][19 + j * 40] = 1 

    # if n == 4:
    #     for i in range(n):
    #         for j in range(n):
    #             ind[39 + i * 40][39 + j * 40] = 1 

    # 400*400分辨率
    if n == 4:
        for i in range(n):
            for j in range(n):
                ind[49 + i * 100][49 + j * 100] = 1 
    #     ind[39][39] = 0
        # for i in range(n * n):
        #     ind[random.randint(0,199)][random.randint(0,199)] = 1

    # if n == 4:
    #     for i in range(n):
    #         for j in range(n):
    # #             # ind[24 + i * 50-1][24 + j * 50] = 1
    # #             # ind[24 + i * 50][24 + j * 50-1] = 1
    #             ind[24 + i * 50][24 + j * 50] = 1   #中心点
    # #             # ind[24 + i * 50+1][24 + j * 50] = 1
    # #             # ind[24 + i * 50][24 + j * 50+1] = 1

    # if n == 4:
    #     for i in range(n):
    #         for j in range(n):
    # #             # ind[24 + i * 50-1][24 + j * 50] = 1
    # #             # ind[24 + i * 50][24 + j * 50-1] = 1
    #             ind[21 + i * 50][21 + j * 50] = 1   #中心点
    #             # ind[24 + i * 50+1][24 + j * 50] = 1
    # ind[21][21] = 0

    if n == 3:
        for i in range(n):
            for j in range(n):
                ind[34 + i * 66][34 + j * 66] = 1 

    if n == 2:
        for i in range(n):
            for j in range(n):
                ind[49 + i * 100][49 + j * 100] = 1 

    if n == 1:
        ind[100][100] = 1 

    plt.imshow(ind)
    plt.savefig('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_4_refine.png',bbox_inches='tight', pad_inches=0.3)
    ind = np.expand_dims(ind, axis=0)
    # print('n = ', n, 'ind', ind.shape)
    # torch.save(ind, './ind_c1_' + str(n) + '_2.pt')
    torch.save(ind, '/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_4_refine.pt')
    return ind


def make_cnd():
    '''
    制作组件温度提取遮罩
    '''
    
    data = loadmat("/mnt/share1/pengxingwen/Dataset/vp/vp_c1_20k_refine/Example0.mat")
    Fc = torch.from_numpy(data["F"])                                       #组件布局功率
    ones = torch.ones_like(Fc)
    zeros = torch.zeros_like(Fc)
    cnd = torch.where(Fc > 0, ones, zeros) 
    # cnd = cnd.unsqueeze(dim=0)
    # cnd = cnd.numpy() 
    # cnd = cnd.reshape(400, 400)  
    # cnd = cnd.to(device) 
    # print('cnd:', cnd.shape, type(cnd))
    # plt.imshow(cnd, cmap='jet')
    # plt.colorbar()
    # plt.savefig('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/cnd_4_refine.png',bbox_inches='tight', pad_inches=0.3)
    torch.save(cnd, '/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/cnd_4_refine.pt')
# cnd = torch.load('./cnd.pt')
# print('cnd:', cnd.shape, type(cnd))


def get_ind():
    '''
    提取遮罩并保存
    '''

    path = '/mnt/share1/pengxingwen/Dataset/temp_rec/vp/vp_10_c3_random_2_4ob/' 
    test_dataset = MyNewDataset(path + "test_F.npy", path + "test_u.npy")
    batch_size = 1
    test_iter = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=4)
    for i, data2 in enumerate(test_iter):
        X2, y2 = data2        
        X2 = X2.numpy()
        X2 = X2.reshape(200, 200)
        print('X2:', type(X2), X2.shape)
        ind_random = np.where(X2>0, 1, X2)
        torch.save(ind_random, './ind_c3_random_2.pt')
        plt.imshow(ind_random, cmap='jet')
        plt.colorbar()
        plt.savefig('c3_ind_random_2.png',bbox_inches='tight', pad_inches=0.3)

        if i == 0:
            break

# ind_10 = np.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/utils/16-observation_points.npy') 
# ind_20 = torch.from_numpy(observation_points)  
# print('ind_16 :', type(ind_16), ind_16.shape)                                                    #监测点均匀采样密度 n*n

ind_4 = make_ind(4)
# cnd = make_cnd()

# for i in range(100):
#     sample_plot('vp_10_sp', '4*4ob', 20+i, ind_4, None)                    #绘制样本

# sample_plot('vp_c3_55k', '16ob', 0, ind_16, None)                        #绘制样本
# sample_plot('vp_c1_60k', '16ob', 0, ind_16, None)                    #绘制样本


# make_dataset('vp_10_c3_2w', 'vp_10_c3_ts_2k_4ob', ind_4, None)             #提取数据集    
