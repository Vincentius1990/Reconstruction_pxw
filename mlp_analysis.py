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


def cbcolor(p):
    '''
    colorbar setting
    '''
    cb = plt.colorbar(fraction=0.045, pad=0.05)
    cb.ax.tick_params(labelsize=12, rotation=0)
    med = p.max() - p.min()
    cb.set_ticks([p.min(), p.min() + 0.2 * med, p.min() + 0.4 * med, p.min() + 0.6 * med, p.min() + 0.8 * med, p.max()])  #np.max(p)
    cb.update_ticks()


def fig_plot3(a, b, plt_title, plt_item, logname, suffix, n, ind):
    '''
    plot
    '''

    plt.figure(figsize=(a, b))                         
    xa = np.linspace(0, 0.1, xw)
    ya = np.linspace(0, 0.1, yh) 
    x_ob, y_ob = np.nonzero(ind)
    x_ob, y_ob = np.array(x_ob)* 0.1 / xw, np.array(y_ob)* 0.1 / yh

    for i, p in enumerate(plt_item):
        plt.subplot(1, 3, i + 1)     
        plt.tight_layout()
        plt.title(plt_title[i])  
        p = p.cpu().numpy()
        p = p.reshape(xw,yh)
        
        if i == 0:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet')   
            cbcolor(p)
            plt.scatter(y_ob, x_ob, s=20, color='white', marker='s')       
        elif i == 1:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='jet') 
            cbcolor(p)
        elif i == 2:
            plt.contourf(xa, ya, p, alpha = 0.9, levels=150, cmap='YlOrRd')  # 'gray' 'YlOrRd'                   
            cbcolor(p)

        ax = plt.gca()
        ax.set_aspect(1)             
        plt.grid(c='k', ls='-', alpha=0.1)                  
        plt.tick_params(labelsize=10, rotation=0)    
        plt.axis('on')
    plt.savefig('/mnt/share1/pengxingwen/reconstruction_pxw/zip/'+ str(logname) + '/' + str(logname) + '_' + str(suffix) + '-' + str(n) + '.png',bbox_inches='tight', pad_inches=0.3)  


def analysis(casename, logname1, logname2, datasetname, n, ind):
    '''
    result analysis
    '''

    ind = np.array(ind).reshape(200, 200)
    if not os.path.exists('/mnt/share1/pengxingwen/reconstruction_pxw/zip/' + str(logname1)):
        os.makedirs('/mnt/share1/pengxingwen/reconstruction_pxw/zip/' + str(logname1)) 

    tfr_path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(casename) + '/' + str(logname1) + '.pt'
    tfr_path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(casename) + '/' + str(logname1) + '_params.pt'
    model = torch.load(tfr_path1)                           
    model.load_state_dict(torch.load(tfr_path2))           
    # print(tfr_unet, tfr_unet.state_dict())
    # print(tfr_unet, tfr_unet.state_dict())
    
    data = loadmat('/mnt/share1/pengxingwen/Dataset/vp/' + str(datasetname) +'/Example' + str(n) + '.mat')
    layout, u = data["F"], data["u"]                     
    monitor_y = u * ind                                  

    monitor_y, u = monitor_y.reshape(1, 1, 200, 200), u.reshape(1, 1, 200, 200)
    layout, monitor_y, u = torch.tensor(layout).float(), torch.tensor(monitor_y).float(), torch.tensor(u).float()
    # print('layout:', layout.shape, 'monitor_y:', monitor_y.shape, 'u:', u.shape)
    # layout = torch.where(layout==0, monitor_y, layout)

    if logname2 != None:
        mlp_path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/vp/MLP/' + str(logname2) + '.pt'
        mlp_path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/vp/MLP/' + str(logname2) + '_params.pt'
        mlp = torch.load(mlp_path1)                         
        mlp.load_state_dict(torch.load(mlp_path2))           
        # print(mlp, mlp.state_dict())
        
        observation_y = monitor_y[monitor_y != 0]                               
        u_area = loadmat('/mnt/share1/pengxingwen/Dataset/vp/' + str(datasetname) +'/Example' + str(n) + '.mat')['u'][0:2, 86:112]  
        u_area = u_area.reshape(52)
        observation_y, u_area = torch.tensor((observation_y - 298) / 50).float(), torch.tensor((u_area - 298) / 50).float() 
        # observation_y, u_area = (observation_y.clone().detach() - 298) / 50, (u_area.clone().detach() - 298) / 50

    with torch.no_grad(): 
        monitor_y, u = monitor_y.to(device), u.to(device)
        output = model(monitor_y)                
        
        if logname2 != None:
            observation_y, u_area = observation_y.to(device), u_area.to(device)
            heatsink_y = mlp(observation_y)           
            heatsink_y = (heatsink_y * 50) + 298

            heatsink_y = heatsink_y.reshape(2, 26)
            # torch.save(heatsink_y, 'heatsink_y.pt')
            output[0, 0, 0:2, 86:112] = heatsink_y    
            # torch.save(output, 'output.pt')

        # sio.savemat('heatsink.mat', {"u_pre": output.squeeze().cpu().numpy()})
        abs_err = torch.abs(output - u)                    
        # r_err = torch.div(output, u) * 100                  
        # print('output:', output.shape, 'abs_err:', abs_err.shape)

    # plt_item = [u, output, abs_err]                
    # plt_title = ['Real Temperature Field(K) \n and Observation Points (white squares)','Reconstructed Temperature Field(K)','Absolute Error(K)']
    # _a, _b = 17, 5    
    # fig_plot3(_a, _b, plt_title, plt_item, logname1, logname2, n, ind) 

                                  

if __name__ == '__main__':

    xw, yh = 200, 200
    if not os.path.exists('/mnt/share1/pengxingwen/reconstruction_pxw/zip'):
        os.makedirs('/mnt/share1/pengxingwen/reconstruction_pxw/zip') 

    ind_4 = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_4.pt')

    '''Case 1'''
    for i in [16000, 16007, 17007, 16132, 18365, 19900]:        
        analysis('vp_10_c1_4ob_UNetV2', 'vp_10_c1_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'vp_c1_60k', i, ind_4)
