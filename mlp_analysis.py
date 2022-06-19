import os
from termios import VMIN
import torch
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from src.data.load_response import *
from src.data.dataset import MyNewData
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 4*4 均匀测点分布
# x_ob = np.array([24, 24, 24, 24, 74, 74, 74, 74, 124, 124, 124, 124, 174, 174 ,174, 174])  * 0.1 / 200
# y_ob = np.array([24, 74, 124, 174, 24, 74, 124, 174, 24, 74, 124, 174, 24, 74, 124, 174])  * 0.1 / 200

# Case1 dpp选点分布
# x_ob = np.array([33, 98, 149, 52, 139, 46,  27,  98,  157, 176, 0, 199, 0,   100, 0, 199])  * 0.1 / 200
# y_ob = np.array([38, 14, 46,  71, 129, 146, 171, 178, 164, 148, 0, 0,   199, 199, 109, 162])  * 0.1 / 200


def cbcolor(p):
    '''
    颜色条设置
    '''
    cb = plt.colorbar(fraction=0.045, pad=0.05)
    cb.ax.tick_params(labelsize=12, rotation=0)
    med = p.max() - p.min()
    cb.set_ticks([p.min(), p.min() + 0.2 * med, p.min() + 0.4 * med, p.min() + 0.6 * med, p.min() + 0.8 * med, p.max()])  #np.max(p)
    cb.update_ticks()

def cbcolor_error(p):
    '''
    颜色条设置
    '''
    cb = plt.colorbar(fraction=0.045, pad=0.05)
    cb.ax.tick_params(labelsize=20, rotation=0)
    # med = np.max(p) - np.min(p)
    # cb.set_ticks([np.min(p), np.min(p) + 0.2 * med, np.min(p) + 0.4 * med, np.min(p) + 0.6 * med, np.min(p) + 0.8 * med, np.max(p)])  #np.max(p)
    pmax, pmin = 0.4, 0
    med = pmax - pmin
    cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]) 
    cb.update_ticks()

def fig_plot(a, b, plt_title, plt_item, logname, suffix, n, ind):
    '''
    绘图
    '''

    fig = plt.figure(figsize=(a, b))             #绘制样本采用图幅尺寸               
    # fig.suptitle('loss = MAEloss, model = '+str(logname)+str(suffix))    
    xa = np.linspace(0, 0.1, 200)
    ya = np.linspace(0, 0.1, 200) 
    # 测点位置坐标
    x_ob, y_ob = np.nonzero(ind)
    x_ob, y_ob = np.array(x_ob)* 0.1 / 200, np.array(y_ob)* 0.1 / 200

    for i, p in enumerate(plt_item):

        p = p.reshape(200,200)
        p = p.cpu().numpy()
        plt.subplot(1, len(plt_item), i + 1)     #绘制结果
        plt.tight_layout()
        plt.title(plt_title[i], fontsize=20, pad=16)  # , fontsize=16
        levels = np.arange(0, 0.5, 0.1)
        
        if i == 1:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet')        # 热力图 cmap='jet' 
            cbcolor(p)
            plt.scatter(y_ob, x_ob, s=20, color='white', marker='s')       # 散点,表示测点
        if i == 0:
            # plt.contourf(xa, ya, plt_item[1], alpha = 0.3, levels=150, cmap='jet')
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='summer')  # 灰度图, cmap='gray'  cmap='nipy_spectral'云图, cmap='YlOrRd'           
            cbcolor(p)  
        elif i == 3:
            # contourf_ = plt.contourf(xa, ya, p, alpha = 0.9, levels=150, vmin=0, vmax=0.5)  # 灰度图, cmap='gray'  cmap='nipy_spectral'云图, cmap='YlOrRd'                   
            contourf_ = plt.imshow(p, cmap='YlOrRd', origin='lower', vmin=0, vmax=0.5)  # 灰度图, cmap='gray'  cmap='nipy_spectral'云图, cmap='YlOrRd'                   
            plt.xticks(np.arange(0, 240, 40), ['0.00', '0.02', '0.04', '0.06', '0.08', '0.10'])
            plt.yticks(np.arange(0, 240, 40), ['0.00', '0.02', '0.04', '0.06', '0.08', '0.10'])
            cbcolor_error(p)
        elif i == 2:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet') # 热力图 cmap='jet' , cmap='nipy_spectral'云图,
            cbcolor(p)

        # elif i == 1:
        #     plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet') # 热力图 cmap='jet' , cmap='nipy_spectral'云图,
        #     cbcolor(p)
        plt.axis('on')
        # plt.legend(fontsize = 20)  
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(pad=16)
        # frame = plt.gca()
        # # y 轴可见
        # frame.axes.get_yaxis().set_visible(False)
        # # x 轴可见
        # frame.axes.get_xaxis().set_visible(False)

    plt.savefig('/mnt/share1/pengxingwen/reconstruction_pxw/zip/'+str(logname) + '_' + str(suffix) + '-' + str(n) + '.png',bbox_inches='tight', pad_inches=0.3)  


def fig_plot2(a, b, plt_title, plt_item, logname, suffix, n, ind):
    '''
    2图, 绘图
    '''

    plt.figure(figsize=(a, b))             #绘制样本采用图幅尺寸               
    xa = np.linspace(0, 0.1, xw)
    ya = np.linspace(0, 0.1, yh) 
    X, Y = np.meshgrid(xa, ya)
    # 测点位置坐标
    x_ob, y_ob = np.nonzero(ind)   
    x_ob, y_ob = np.array(x_ob) * 0.1 / xw, np.array(y_ob) * 0.1 / yh

    for i, p in enumerate(plt_item):
        plt.subplot(1, 1, i + 1)     #绘制结果
        plt.title(plt_title[i])  
        p = p.cpu().numpy()
        p = p.reshape(xw, yh)
        
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
        # ax.set_aspect('equal', 'box')              # 设置X, Y方向长度相等 
        plt.grid(c='k', ls='-', alpha=0.1)           # 显示网格                    
        plt.tick_params(labelsize=12, rotation=0)    # 设置刻度字体大小和旋转角度
        plt.axis('on')
        # plt.xticks(fontsize=13)
        # plt.yticks(fontsize=13)
        # ax.axes.get_yaxis().set_visible(False)  # y 轴可见
        # ax.axes.get_xaxis().set_visible(False)  # x 轴可见
    plt.savefig('/mnt/share1/pengxingwen/reconstruction_pxw/zip/'+ str(logname) + '/' + str(logname) + '_' + str(suffix) + '-' + str(n) + '.png',bbox_inches='tight', pad_inches=0.3)  


def fig_plot4(a, b, plt_title, plt_item, logname, suffix, n, ind):
    '''
    四图, 绘图
    '''

    plt.figure(figsize=(a, b))             #绘制样本采用图幅尺寸               
    xa = np.linspace(0, 0.1, xw)
    ya = np.linspace(0, 0.1, yh) 
    X, Y = np.meshgrid(xa, ya)
    # 测点位置坐标
    x_ob, y_ob = np.nonzero(ind)   
    x_ob, y_ob = np.array(x_ob) * 0.1 / xw, np.array(y_ob) * 0.1 / yh

    for i, p in enumerate(plt_item):
        plt.subplot(1, 4, i + 1)     #绘制结果
        plt.title(plt_title[i])  
        p = p.cpu().numpy()
        p = p.reshape(xw, yh)
        
        if i == 0:
            plt.contourf(X, Y, p, alpha = 0.9, levels=150, cmap='summer')   # jet, seismic summer
            # C = plt.contour(X, Y, p, 10, colors='black', linewidths=0.5, alpha=0.8)  # 画出等高线
            # plt.clabel(C, inline=1, fontsize=12)  # 标出温度值
            cbcolor(p)
        elif i == 1:
            plt.contourf(X, Y, p, alpha = 0.9, levels=150, cmap='jet')   # jet, seismic
            cbcolor(p)
            plt.scatter(x_ob, y_ob, s=20, color='white', marker='s')     # 散点,表示测点
        elif i == 2:
            plt.contourf(X, Y, p, alpha = 0.9, levels=150, cmap='jet')   # jet, nipy_spectral
            cbcolor(p)
        elif i == 3:
            plt.contourf(X, Y, p, alpha = 0.9, levels=150, cmap='YlOrRd')  # YlOrRd, gray, nipy_spectral                  
            cbcolor(p)

            # contourf_ = plt.imshow(p, cmap='YlOrRd', origin='lower', vmin=0, vmax=0.5)  # 灰度图, cmap='gray'  cmap='nipy_spectral'云图, cmap='YlOrRd'                   
            # plt.xticks(np.arange(0, 240, 40), ['0.00', '0.02', '0.04', '0.06', '0.08', '0.10'])
            # plt.yticks(np.arange(0, 240, 40), ['0.00', '0.02', '0.04', '0.06', '0.08', '0.10'])
            # cbcolor_error(p)


        ax = plt.gca()
        ax.set_aspect(1)
        # ax.set_aspect('equal', 'box')              # 设置X, Y方向长度相等 
        plt.grid(c='k', ls='-', alpha=0.1)           # 显示网格                    
        plt.tick_params(labelsize=12, rotation=0)    # 设置刻度字体大小和旋转角度
        plt.axis('on')
        # plt.xticks(fontsize=13)
        # plt.yticks(fontsize=13)
        # ax.axes.get_yaxis().set_visible(False)  # y 轴可见
        # ax.axes.get_xaxis().set_visible(False)  # x 轴可见
    plt.savefig('/mnt/share1/pengxingwen/reconstruction_pxw/zip/'+ str(logname) + '/' + str(logname) + '_' + str(suffix) + '-' + str(n) + '.png',bbox_inches='tight', pad_inches=0.3)  


def fig_plot3(a, b, plt_title, plt_item, logname, suffix, n, ind):
    '''
    三图, 绘图
    '''

    plt.figure(figsize=(a, b))             #绘制样本采用图幅尺寸               
    xa = np.linspace(0, 0.1, xw)
    ya = np.linspace(0, 0.1, yh) 
    # 测点位置坐标
    x_ob, y_ob = np.nonzero(ind)
    x_ob, y_ob = np.array(x_ob)* 0.1 / xw, np.array(y_ob)* 0.1 / yh

    for i, p in enumerate(plt_item):
        plt.subplot(1, 3, i + 1)     #绘制结果
        plt.tight_layout()
        plt.title(plt_title[i])  
        p = p.cpu().numpy()
        p = p.reshape(xw,yh)
        
        if i == 0:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet')   # 热力图 cmap='jet' 
            cbcolor(p)
            plt.scatter(y_ob, x_ob, s=20, color='white', marker='s')       # 散点,表示测点
        elif i == 1:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet') # 热力图 cmap='jet' , cmap='nipy_spectral'云图,
            cbcolor(p)
        elif i == 2:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='YlOrRd')  # 'gray' 'YlOrRd'                   
            cbcolor(p)
            # # cb.set_ticks([0, 0.1, 0.3, 0.5, 0.7])  #np.max(p)
            # cb.update_ticks()

        ax = plt.gca()
        ax.set_aspect(1)
        # ax.set_aspect('equal', 'box')                # 设置X, Y方向长度相等 
        plt.grid(c='k', ls='-', alpha=0.1)           # 显示网格                    
        plt.tick_params(labelsize=10, rotation=0)    # 刻度字体大小和旋转
        plt.axis('on')
        # plt.xticks(fontsize=13)
        # plt.yticks(fontsize=13)
        # ax.axes.get_yaxis().set_visible(False)      # y 轴可见
        # ax.axes.get_xaxis().set_visible(False)      # x 轴可见

    plt.savefig('/mnt/share1/pengxingwen/reconstruction_pxw/zip/'+ str(logname) + '/' + str(logname) + '_' + str(suffix) + '-' + str(n) + '.png',bbox_inches='tight', pad_inches=0.3)  



def analysis(logname, datasetname, ind, n):
    '''
    读取训练好的模型，载入参数
    读取测试数据集，进行分析并绘图
    '''

    tfr_path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(logname[0:2]) + '/' + str(logname) + '.pt'
    tfr_path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(logname[0:2]) + '/' + str(logname) + '_params.pt'
    model = torch.load(tfr_path1)                                #载入模型
    model.load_state_dict(torch.load(tfr_path2))                 #载入参数
    # print(tfr_unet, tfr_unet.state_dict())

    root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'
    test_path = '/mnt/share1/pengxingwen/Dataset/vp/' + datasetname
    batch_size = 1                                    
    test_dataset = MyNewData(root, test_path, ind, None)
    test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=4)
    # print('test_dataset:',len(test_iter))

    for i, batch_data in enumerate(test_iter):                                        
        monitor_y, u = batch_data
        # print('monitor_y:', monitor_y.shape, 'u:', u.shape)
        monitor_y, u = monitor_y.to(device), u.to(device)       
        with torch.no_grad():
            output = model(monitor_y)                        #预测温度场
            abs_err = torch.abs(output - u)                  #温度场绝对误差
            r_err = torch.div(output, u) * 100               #温度场相对误差
     
        plt_item = [monitor_y, output, u, abs_err]           #真实功率布局,监测点温度,组件温度,真实温度场
        plt_title = ['Observation Points(K)','Reconstructed Temperature Field(K)','Real Temperature Field(K)','Absolute Error(K)']
        _c, _d = 23, 5
        fig_plot(_c, _d, plt_title, plt_item, logname, '_lack', i) 
        if i == n:
            break          
# analysis('vp_c1_ts_2k_4ob_UNetV2_200epoch', 'test_list_5k.txt', ind_4, 1) 

def paper(casename, logname1, logname2, datasetname, n, ind):
    '''
    撰写论文绘图，同时呈现热源布局、监测点、重建温度场、真实温度场、绝对误差和相对误差
    读取训练好的模型，载入参数
    读取指定样本，重建温度场，分析
    '''

    ind = np.array(ind).reshape(200, 200)
    if not os.path.exists('/mnt/share1/pengxingwen/reconstruction_pxw/zip/' + str(logname1)):
        os.makedirs('/mnt/share1/pengxingwen/reconstruction_pxw/zip/' + str(logname1)) 

    # 读取UNet模型
    tfr_path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(casename) + '/' + str(logname1) + '.pt'
    tfr_path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(casename) + '/' + str(logname1) + '_params.pt'
    model = torch.load(tfr_path1)                           #载入模型
    model.load_state_dict(torch.load(tfr_path2))            #载入参数
    # print(tfr_unet, tfr_unet.state_dict())
    # print(tfr_unet, tfr_unet.state_dict())
    
    # 读取第N个样本mat文件
    data = loadmat('/mnt/share1/pengxingwen/Dataset/vp/' + str(datasetname) +'/Example' + str(n) + '.mat')
    layout, u = data["F"], data["u"]                     # 载荷,温度场
    monitor_y = u * ind                                  # 监测点

    # UNet模型输入数据
    monitor_y, u = monitor_y.reshape(1, 1, 200, 200), u.reshape(1, 1, 200, 200)
    layout, monitor_y, u = torch.tensor(layout).float(), torch.tensor(monitor_y).float(), torch.tensor(u).float()
    # print('layout:', layout.shape, 'monitor_y:', monitor_y.shape, 'u:', u.shape)
    # layout = torch.where(layout==0, monitor_y, layout)

    # 读取MLP模型
    if logname2 != None:
        mlp_path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/vp/MLP/' + str(logname2) + '.pt'
        mlp_path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/vp/MLP/' + str(logname2) + '_params.pt'
        mlp = torch.load(mlp_path1)                           #载入模型
        mlp.load_state_dict(torch.load(mlp_path2))            #载入参数
        # print(mlp, mlp.state_dict())
        
        # MLP模型输入数据
        observation_y = monitor_y[monitor_y != 0]            # 提取温度测点数据                          
        u_area = loadmat('/mnt/share1/pengxingwen/Dataset/vp/' + str(datasetname) +'/Example' + str(n) + '.mat')['u'][0:2, 86:112]  # 提取热沉区域30*5温度数据
        u_area = u_area.reshape(52)
        observation_y, u_area = torch.tensor((observation_y - 298) / 50).float(), torch.tensor((u_area - 298) / 50).float() 
        # observation_y, u_area = (observation_y.clone().detach() - 298) / 50, (u_area.clone().detach() - 298) / 50

    with torch.no_grad(): 
        monitor_y, u = monitor_y.to(device), u.to(device)
        output = model(monitor_y)                   # UNet预测温度场
        
        if logname2 != None:
            observation_y, u_area = observation_y.to(device), u_area.to(device)
            heatsink_y = mlp(observation_y)            # MLP预测温度场
            heatsink_y = (heatsink_y * 50) + 298

            heatsink_y = heatsink_y.reshape(2, 26)
            # torch.save(heatsink_y, 'heatsink_y.pt')
            output[0, 0, 0:2, 86:112] = heatsink_y     # 热沉区域回填
            # torch.save(output, 'output.pt')

        # sio.savemat('heatsink.mat', {"u_pre": output.squeeze().cpu().numpy()})
        abs_err = torch.abs(output - u)                     #温度场绝对误差
        # r_err = torch.div(output, u) * 100                  #温度场相对误差
        # print('output:', output.shape, 'abs_err:', abs_err.shape)

    # plt_item = [layout, u, output, abs_err]                #真实功率布局,监测点温度,组件温度,真实温度场
    # plt_title = ['Layout and Power Intensities(W/m$^2$)', 'Real Temperature Field(K) and \n Observation Points (white squares)', 'Reconstructed Temperature Field(K)', 'Absolute Error(K)']
    # _a, _b = 26, 5.6   # 四图

    plt_item = [u]                #真实功率布局,监测点温度,组件温度,真实温度场
    plt_title = ['Real Temperature Field(K)']
    _a, _b = 7 , 5.6    # 一图
    fig_plot2(_a, _b, plt_title, plt_item, logname1, logname2, n, ind) 

    # plt_item = [u, output, abs_err]                #真实功率布局,监测点温度,组件温度,真实温度场
    # plt_title = ['Real Temperature Field(K) \n and Observation Points (white squares)','Reconstructed Temperature Field(K)','Absolute Error(K)']
    # _a, _b = 17, 5    # 三图
    # fig_plot3(_a, _b, plt_title, plt_item, logname1, logname2, n, ind) 

    # plt_item = [layout, u, output, abs_err]                #真实功率布局,监测点温度,组件温度,真实温度场
    # plt_title = ['Layout and Power Intensities(W/m$^2$)', 'Real Temperature Field(K)', 'Reconstructed Temperature Field(K)', 'Absolute Error(K)']
    # _a, _b = 30 , 5.6   # 四图
    # fig_plot4(_a, _b, plt_title, plt_item, logname1, logname2, n, ind) 
                                  

if __name__ == '__main__':
    '''
    主函数,读取测试集,测试不同模型的效果
    '''
    xw, yh = 200, 200
    if not os.path.exists('/mnt/share1/pengxingwen/reconstruction_pxw/zip'):
        os.makedirs('/mnt/share1/pengxingwen/reconstruction_pxw/zip') 
    # ind_3 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_3.pt')
    # ind_5 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_5.pt')

    ind_4 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_4.pt')
    # ind_4_2 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_4_2.pt')
    # ind_4_2 = ind_4_2.reshape(1, 1, 200 ,200)
    # ind_4_3 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_4_3.pt')
    # ind_c1_16 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_c1_16.pt')
    # ind_c1_random_1 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_c1_random_1.pt')
    # ind_c1_random_2 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_c1_random_2.pt')
    # ind_c1_random_3 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_c1_random_3.pt')
    # ind_2 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_2.pt')

    # ind_c3_16 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_c3_16.pt')
    # ind_c3_random_1 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_c3_random_1.pt')
    # ind_c3_random_2 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_c3_random_2.pt')

    '''Case 1''' # 16000, 16010, 16115, 17000, 17001, 18000, 19000, 18001, 19001, 19999 20, 1, 10, 100, 500, 600, 60, 666, 888, 999, 1000, 1024, 1025
    for i in [16000, 16007, 17007, 16132, 18365, 19900]:
        # paper('vp_10_c1_4ob_UNetV2', 'vp_10_c1_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_sp', i, ind_4)
        # paper('vp_10_c1_0.1grad_4ob_UNetV2', 'vp_10_c1_0.1grad_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_10_c1_0.1TV_4ob_UNetV2', 'vp_10_c1_0.1TV_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_10_c1_0.1TV_4ob_UNetV2', 'vp_10_c1_0.1TV_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_sp', i, ind_4)
        # paper('vp_10_c1_5ob_UNetV2', 'vp_10_c1_5ob_UNetV2_200epoch', 'vp_10_c1_5ob_MLP_100epoch', 'vp_c1_60k', n, ind_5)
        
        paper('vp_10_c1_4ob_UNetV2', 'vp_10_c1_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_c1_4ob_0.1grad_UNetV2', 'vp_c1_4ob_0.1grad_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_c1_4ob_0.2grad_UNetV2', 'vp_c1_4ob_0.2grad_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_c1_4ob_0.3grad_UNetV2', 'vp_c1_4ob_0.3grad_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_c1_4ob_0.4grad_UNetV2', 'vp_c1_4ob_0.4grad_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_c1_4ob_0.5grad_UNetV2', 'vp_c1_4ob_0.5grad_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_c1_4ob_0.6grad_UNetV2', 'vp_c1_4ob_0.6grad_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_c1_4ob_0.8grad_UNetV2', 'vp_c1_4ob_0.8grad_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
        # paper('vp_c1_4ob_1.0grad_UNetV2', 'vp_c1_4ob_1.0grad_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)

        
    

    # paper('vp_10_c1_16ob_UNetV2_200epoch_2', 'vp_10_c1_16ob_MLP_100epoch', 'vp_c1_60k', n, ind_c1_16)

    # paper('vp_10_c1_random_1_4ob_UNetV2_200epoch', 'vp_10_c1_random_1_4ob_MLP_100epoch', 'vp_c1_60k', n, ind_c1_random_1)
    # paper('vp_10_c1_random_2_4ob_UNetV2_200epoch', 'vp_10_c1_random_2_4ob_MLP_100epoch', 'vp_c1_60k', n, ind_c1_random_2)
    # paper('vp_10_c1_random_3_4ob_UNetV2_200epoch', 'vp_10_c1_random_3_4ob_MLP_100epoch', 'vp_c1_60k', n, ind_c1_random_3)
    
    # paper('vp_10_c1_0.1TV_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_sp', 20, ind_4)
    # paper('vp_10_c1_0.1TV_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_sp', 1024, ind_4)
    # paper('vp_10_c1_0.1TV_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_sp', 1025, ind_4)

    # paper('vp_10_c1_2ob_UNetV2_200epoch', 'vp_10_c1_2ob_MLP_100epoch', 'vp_c1_60k', n, ind_2)

    '''Case 2'''  # 16003, 16010, 16115, 17000, 17001, 18000, 19000, 18001, 19001, 19999
    # for i in [1, 12, 100, 150, 200, 500, 800, 666, 1020, 1026, 1027]:
        # paper('vp_c3_4ob_UNetV2', 'vp_c3_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'vp_c3_55k', i, ind_4)
        # paper('vp_c3_0.1grad_4ob_UNetV2', 'vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'vp_c3_55k', i, ind_4)
        # paper('vp_c3_0.1grad_4ob_UNetV2', 'vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'vp_c3_sp', i, ind_4)
        # paper('vp_c3_0.1TV_4ob_UNetV2', 'vp_c3_0.1TV_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'vp_c3_55k', i, ind_4)

    # paper('vp_c3_16ob_UNetV2_200epoch', 'vp_c3_16ob_MLP_100epoch', 'vp_c3_55k', n, ind_c3_16)

    # paper('vp_c3_random_1_4ob_UNetV2_200epoch', 'vp_c3_random_1_4ob_MLP_100epoch', 'vp_c3_55k', n, ind_c3_random_1)
    # paper('vp_c3_random_2_4ob_UNetV2_200epoch', 'vp_c3_random_2_4ob_MLP_100epoch', 'vp_c3_55k', n, ind_c3_random_2)

    # paper('vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'vp_c3_55k', 16003, ind)
    # paper('vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'vp_c3_sp', 12, ind_4)
    # paper('vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'vp_c3_sp', 1026, ind_4)
    # paper('vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'vp_c3_sp', 1027, ind_4)

    # for i in range(10):
        # paper('vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_30epoch', 'vp_c3_55k', 16000+i, ind)
        # paper('vp_c3_0.1TV_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_30epoch', 'vp_c3_55k', 16000+i, ind)
        # paper('vp_c3_4ob_UNetV2_200epoch_4', 'vp_c3_4ob_MLP_30epoch', 'vp_c3_55k', 16000+i, ind)
        # paper('vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_30epoch', 'vp_c3_sp', i, ind)

        # paper('vp_10_c1_0.1grad_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_30epoch', 'vp_c1_60k', 16000+i, ind)
        # paper('vp_10_c1_0.1TV_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_30epoch', 'vp_c1_sp', 20+i, ind)
        # paper('vp_10_c1_4ob_UNetV2_200epoch', None, 'vp_c1_60k', 16000+i, ind)

        # paper('vp_10_c1_pd_16ob_UNetV2_200epoch', 'vp_10_c1_pd_16ob_MLP_100epoch', 'vp_c1_60k', 16000+i, ind)

        # paper('vp_10_c1_ts_10k_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_30epoch', 'vp_c1_60k', 16000+i, ind)

        # paper('vp_10_c1_random_3_4ob_UNetV2_200epoch', 'vp_10_c1_random_3_4ob_MLP_100epoch', 'vp_c1_60k', 16000+i, ind)
