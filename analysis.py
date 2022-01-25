import os
import torch
import numpy as np
from pathlib import Path
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from src.data.load_response import *
from src.data.dataset import MyNewData, make_dataset
from src.data.datapre import MyDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_point(monitor_y, layout_pred):
    '''
    提取监测点并比较
    '''
    
    ones = torch.ones_like(monitor_y).cuda()
    zeros = torch.zeros_like(monitor_y).cuda()
    ind = torch.where(monitor_y < 267, ones, zeros)   
    return (layout_pred * ind)


def cbcolor(p):
    '''
    颜色条设置
    '''
    cb = plt.colorbar(fraction=0.045, pad=0.05)
    med = np.max(p) - np.min(p)
    cb.set_ticks([np.min(p), np.min(p) + 0.2 * med, np.min(p) + 0.4 * med, np.min(p) + 0.6 * med, np.min(p) + 0.8 * med, np.max(p)])  #np.max(p)
    cb.update_ticks()


def fig_plot(a, b, plt_title, plt_item, logname, suffix, n):
    '''
    绘图
    '''

    fig = plt.figure(figsize=(a, b))             #绘制样本采用图幅尺寸               
    # fig.suptitle('loss = MAEloss, model = '+str(logname)+str(suffix))    
    xa = np.linspace(0, 0.1, 200)
    ya = np.linspace(0, 0.1, 200)   
    for i, p in enumerate(plt_item):
        p = p.reshape(200,200)
        p = p.cpu().numpy()
        plt.subplot(1, len(plt_item), i + 1)     #绘制结果
        plt.tight_layout()
        plt.title(plt_title[i])
        # plt.imshow(p, cmap='jet')              #cmap='jet', 'nipy_spectral' 热力图，云图
        plt.contourf(xa, ya, p, levels=150, cmap='jet')        
        plt.axis('on')

        cb = plt.colorbar(fraction=0.045, pad=0.05)
        med = np.max(p) - np.min(p)
        cb.set_ticks([np.min(p), np.min(p) + 0.2 * med, np.min(p) + 0.4 * med, np.min(p) + 0.6 * med, np.min(p) + 0.8 * med, np.max(p)])  #np.max(p)
        cb.update_ticks()

    plt.savefig('./zip/'+str(logname) + str(suffix) + '-' + str(n) + '.png',bbox_inches='tight', pad_inches=0.3)  


def analysis(logname, datasetname, ind, n):
    '''
    读取训练好的模型，载入参数
    读取测试数据集，进行分析并绘图
    '''

    tfr_path1 = './results/' + str(logname[0:2]) + '/' + str(logname) + '.pt'
    tfr_path2 = './results/' + str(logname[0:2]) + '/' + str(logname) + '_params.pt'
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


def paper(logname, datasetname, n, ind):
    '''
    撰写论文绘图，同时呈现热源布局、监测点、重建温度场、真实温度场、绝对误差和相对误差
    读取训练好的模型，载入参数
    读取指定样本，重建温度场，分析
    '''

    tfr_path1 = './results/' + str(logname[0:2]) + '/' + str(logname) + '.pt'
    tfr_path2 = './results/' + str(logname[0:2]) + '/' + str(logname) + '_params.pt'
    model = torch.load(tfr_path1)                           #载入模型
    model.load_state_dict(torch.load(tfr_path2))            #载入参数
    # print(tfr_unet, tfr_unet.state_dict())
    
    data = loadmat('/mnt/share1/pengxingwen/Dataset/vp/' + str(datasetname) +'/Example' + str(n) + '.mat')
    layout, u = data["F"], data["u"]                        #载荷,温度场
    monitor_y = u * ind                                     #监测点

    # layout = layout + monitor_y*20
    monitor_y, u = monitor_y.reshape(1,1,200,200), u.reshape(1,1,200,200)
    layout, monitor_y, u = torch.tensor(layout), torch.tensor(monitor_y).float(), torch.tensor(u).float()
    # print('layout:', layout.shape, 'monitor_y:', monitor_y.shape, 'u:', u.shape)
    # layout = torch.where(layout==0, monitor_y, layout)

    with torch.no_grad(): 
        monitor_y, u = monitor_y.to(device), u.to(device)
        output = model(monitor_y)                           #预测温度场
        sio.savemat('heatsink.mat', {"u_pre": output.squeeze().cpu().numpy()})
        abs_err = torch.abs(output - u)                     #温度场绝对误差
        r_err = torch.div(output, u) * 100                  #温度场相对误差
        # print('output:', output.shape, 'abs_err:', abs_err.shape)

    plt_item = [layout, monitor_y, output, u, abs_err]                #真实功率布局,监测点温度,组件温度,真实温度场
    plt_title = ['Power(W)','Observation Points(K)','Reconstructed Temperature Field(K)','Real Temperature Field(K)','Absolute Error(K)']
    _a, _b = 29, 5   # 五图
    
    # plt_item = [monitor_y, output, abs_err]                #真实功率布局,监测点温度,组件温度,真实温度场
    # plt_title = ['Observation Points(K)','Reconstructed Temperature Field(K)','Absolute Error(K)']
    # _a, _b = 17, 5    # 三图

    fig_plot(_a, _b, plt_title, plt_item, logname, '', n)           

# analysis('vp_c1_ts_2k_4ob_UNetV2_200epoch', 'test_list_5k.txt', ind_4, 1) 

if not os.path.exists('zip'):
    os.makedirs('zip') 
ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_4_3.pt')
for i in range(3):
    # paper('vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_55k', 16000+i, ind)
    # paper('vp_c3_4ob_UNetV2_200epoch', 'vp_c3_55k', 16000+i, ind)
