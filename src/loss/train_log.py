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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def train_log_draw(dataset, model, num_epochs):

    if str(model) == 'MLP':
        train_log = np.load('/mnt/share1/pengxingwen/reconstruction_pxw/results/runs/' \
    + str(dataset[0:2]) + '/MLP/'+ str(dataset) + '_' + str(model)+ '_' + str(num_epochs) + 'epoch/train_log.npy')
        path = 'results/runs/' + str(dataset[0:2]) + '/MLP/'+ str(dataset) + '_'\
     + str(model)+ '_' + str(num_epochs) + 'epoch/Train_curve.png'
    else:
        train_log = np.loadtxt('/mnt/share1/pengxingwen/reconstruction_pxw/results/' \
    + str(dataset) + '_' + str(model) + '/train_log' + '_' + str(num_epochs) + 'epoch.txt')
        path = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model) \
    + '/Train_curve' + '_' + str(num_epochs) + 'epoch.png'
    
    print('train_log: \nTrain loss = {:.6f} \nTest loss = {:.6f} \nMAE = {:.6f} \nCMAE = {:.6f}K' 
    '\nMaxAE = {:.6f}K \nMT-AE = {:.6f}K '.format(train_log[-1][0], \
    train_log[-1][1], train_log[-1][2], train_log[-1][3], train_log[-1][4], train_log[-1][5]))

    x_epoch = range(num_epochs-1)
    fig = plt.figure(figsize=(20, 24))
    plt.subplot(3, 1, 1)  
    plt.title('Train Curve: Train loss = {:.4f}, Test loss = {:.4f}'\
    .format(train_log[-1][0], train_log[-1][1]))   
    plt.plot(x_epoch, train_log[:,0], 'r', label=u'Train loss')
    plt.plot(x_epoch, train_log[:,1], 'b', label=u'Test loss')
    plt.xlabel("Epochs")                              
    plt.ylabel("Loss")                               
    plt.legend()                                     

    plt.subplot(3, 1, 2) 
    plt.title('Error Curve: MAE = {:.4f}, CMAE = {:.4f}'\
    .format(train_log[-1][2], train_log[-1][3]))
    plt.plot(x_epoch, train_log[:,2], 'g', label=u'MAE')
    plt.plot(x_epoch, train_log[:,3], 'c', label=u'CMAE') 
    plt.xlabel("Epochs")                             
    plt.ylabel("Error")                              
    plt.legend()                                    

    plt.subplot(3, 1, 3) 
    plt.title('Error Curve: MaxAE = {:.4f}, MT-AE = {:.4f}'\
    .format(train_log[-1][4], train_log[-1][5]))
    plt.plot(x_epoch, train_log[:,4], 'g', label=u'MaxAE') 
    plt.plot(x_epoch, train_log[:,5], 'g', label=u'MT-AE') 
    plt.xlabel("Epochs")                             
    plt.ylabel("Error")                             
    plt.legend()                                   

    plt.savefig(path, bbox_inches='tight', pad_inches=0.3) 
