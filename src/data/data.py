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

    ind = ind.reshape(200, 200)                       #遮罩降维
    # monitor_y = u * ind
    print('Fc', Fc.shape, type(Fc))
    print('u',  u.shape, type(u))
    # print('monitor_y',  monitor_y.shape, type(monitor_y))

    # plt_item = [Fc, monitor_y, u]      #真实功率布局,监测点温度,组件温度,真实温度场
    # plt_title = ['Power Intensities (W/m$^2$)', 'Observation Points(K)', 'Real Temperature Field(K)']
    # _a, _b = 23, 5   # 3图 17, 5

    # plt_item = [Fc]                         #真实功率布局
    # plt_title = ['Power Intensities (W/m$^2$)']
    # _a, _b = 16, 14.5   # 1图

    plt_item = [Fc, u]                    #真实功率布局, 真实温度场
    plt_title = ['Power Intensities (W/m$^2$)', 'Real Temperature Field(K)']
    _a, _b = 15 , 5.6   # 2图

    plt_item = [u]                    #真实功率布局, 真实温度场
    plt_title = ['Temperature Field(K) 250x250 grid']
    _a, _b = 7 , 5.6   # 1图
    
    fig_plot2(_a, _b, plt_title, plt_item, dataset_name, obs_name, n, ind)


def cbcolor(p):
    '''
    颜色条设置
    '''
    cb = plt.colorbar(fraction=0.045, pad=0.05)
    cb.ax.tick_params(labelsize=12, rotation=0)
    med = p.max() - p.min()
    cb.set_ticks([p.min(), p.min() + 0.2 * med, p.min() + 0.4 * med, p.min() + 0.6 * med, p.min() + 0.8 * med, p.max()])  #np.max(p)
    cb.update_ticks()


def fig_plot2(a, b, plt_title, plt_item, dataset_name, obs_name, n, ind):
    '''
    1图, 绘图
    '''

    plt.figure(figsize=(a, b))             #绘制样本采用图幅尺寸               
    xa = np.linspace(0, 0.1, xw)
    ya = np.linspace(0, 0.1, yh) 
    X, Y = np.meshgrid(xa, ya)
    # 测点位置坐标
    x_ob, y_ob = np.nonzero(ind)
    x_ob, y_ob = np.array(x_ob)* 0.1 / xw, np.array(y_ob)* 0.1 / yh

    for i, p in enumerate(plt_item):
        plt.subplot(1, 1, i + 1)     #绘制结果
        plt.tight_layout()
        plt.title(plt_title[i])  
        

        '''一图绘制'''
        if i == 0:
            plt.contourf(X, Y, p, alpha = 0.9, levels=150, cmap='jet')   # jet, seismic summer
            # C = plt.contour(X, Y, p, 10, colors='black', linewidths=0.5, alpha=0.8)  # 画出等高线
            # plt.clabel(C, inline=1, fontsize=12)  # 标出温度值
            cbcolor(p)
        
        '''二图绘制'''
        # if i == 0:
        #     plt.contourf(X, Y, p, alpha = 0.9, levels=150, cmap='summer')   # jet, seismic summer
        #     # C = plt.contour(X, Y, p, 10, colors='black', linewidths=0.5, alpha=0.8)  # 画出等高线
        #     # plt.clabel(C, inline=1, fontsize=12)  # 标出温度值
        #     cbcolor(p)
        # elif i == 1:
        #     plt.contourf(X, Y, p, alpha = 0.9, levels=150, cmap='jet')   # jet, seismic
        #     cbcolor(p)
        #     # plt.scatter(x_ob, y_ob, s=20, color='white', marker='s')     # 散点,表示测点

        ax = plt.gca()
        ax.set_aspect(1)
        # ax.set_aspect('equal', 'box')                # 设置X, Y方向长度相等 
        plt.grid(c='k', ls='-', alpha=0.1)           # 显示网格                    
        plt.tick_params(labelsize=12, rotation=0)    # 刻度字体大小和旋转
        plt.axis('on')
        # plt.xticks(fontsize=13)
        # plt.yticks(fontsize=13)
        # ax.axes.get_yaxis().set_visible(False)      # y 轴可见
        # ax.axes.get_xaxis().set_visible(False)      # x 轴可见
    plt.savefig("./sample/sample_" + str(dataset_name) + "_" + str(obs_name) + "-" + str(n) + ".png",bbox_inches='tight', pad_inches=0.3)


def fig_plot(a, b, plt_title, plt_item, dataset_name, obs_name, n):
    '''
    绘图显示结果
    '''

    fig = plt.figure(figsize=(a, b))  # 绘制样本采用图幅尺寸    
    # fig = plt.figure(figsize=(6.8*len(plt_item), 5.4))                # 设置幅面大小,a为图形的宽,b为图形的高,单位为英寸
    # fig.suptitle(str(dataset_name)+'_'+str(obs_name)+' sample')       # 图片标题
    xa = np.linspace(0, 0.1, xw)
    ya = np.linspace(0, 0.1, yh)  
    for i, p in enumerate(plt_item):
        plt.subplot(1, len(plt_item), i + 1)                            # 绘制子图
        plt.title(plt_title[i])                                         #, fontsize=50

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

        ax = plt.gca()
        ax.set_aspect(1)
        # ax.set_aspect('equal', 'box')              # 设置X, Y方向长度相等 
        plt.grid(c='k', ls='-', alpha=0.1)           # 显示网格                    
        plt.tick_params(labelsize=12, rotation=0)    # 设置刻度字体大小和旋转角度
        plt.axis('on')

    if not os.path.exists('sample'):
        os.makedirs('sample') 
    plt.savefig("./sample/sample_" + str(dataset_name) + "_" + str(obs_name) + "-" + str(n) + ".png",bbox_inches='tight', pad_inches=0.3)
   
                           

root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c3_sp'
train_path = '/mnt/share1/pengxingwen/Dataset/vp/train.txt'
test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'
ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_4.pt')

# for i in range(20):
#     sample_plot('vp_c3_55k', '4ob', 16000+i, ind, None)                 #绘制样本

# sample_plot('vp_c1_60k', '4ob', 16000, ind, None)                 #绘制样本
# sample_plot('vp_c3_55k', '4ob', 16003, ind, None)                 #绘制样本

# sample_plot('vp_c1_sp', '4ob', 1024, ind, None)                 #绘制样本
# sample_plot('vp_c1_sp', '4ob', 20, ind, None)                 #绘制样本
# sample_plot('vp_c1_sp', '4ob', 1025, ind, None)                 #绘制样本
# sample_plot('vp_c3_sp', '4ob', 1026, ind, None)                 #绘制样本
# sample_plot('vp_c3_sp', '4ob', 12, ind, None)                 #绘制样本
# sample_plot('vp_c3_sp', '4ob', 1027, ind, None)                 #绘制样本

# xw, yh = 250, 250
# sample_plot('vp_c1_grid', '4ob', 250, ind, None) 

# data = loadmat("/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k/Example16000.mat")
# # print(data.keys() ,data.values())              #输出数据集关键字
# u, Fc = data["u"], data["F"]                     #温度场、监测点和载荷
# Fc = torch.from_numpy(data["F"]) 
# print(Fc, type(Fc))
# ones = torch.ones_like(Fc)
# zeros = torch.zeros_like(Fc)
# cnd = torch.where(Fc == 6616.3068536755, ones, zeros)
# plt.imshow(cnd, origin='lower')
# plt.colorbar()
# plt.savefig('/mnt/share1/pengxingwen/cnd.png',bbox_inches='tight', pad_inches=0.3)
# # np.savetxt('/mnt/share1/pengxingwen/layout_generator/layout16000.txt', Fc)
# power = set(Fc[Fc>0])

data = loadmat("/mnt/share1/pengxingwen/Dataset/vp/vp_c1_grid/Example50.mat")
# # print(data.keys() ,data.values())              #输出数据集关键字
u, Fc = data["u"], data["F"]                     #温度场、监测点和载荷
line_a, line_b = u[24, :], u[:, 24]
print(line_a)
