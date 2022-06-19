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
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.data.dataset import HeatsinkData
from src.models.MLP import MLPV2
from src.loss.train_log import train_log_draw


def train(dataset, model, net, optimizer, device, num_epochs, criterion):
    '''
    train the neural networks
    '''

    train_dataset = HeatsinkData(root, train_path, ind, None)                                            
    train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle= True, num_workers=16)     
    test_dataset = HeatsinkData(root, test_path, ind, None)                                               
    test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)    
    l_train, l_test = train_dataset.__len__(), test_dataset.__len__()                                 
    train_log = np.zeros([num_epochs, 6])                                                            
    
    print('Training on', device, 'model = ' + str(model), 'loss = MAE')             
    print('dataset = ' + str(dataset), 'train_set samples:', l_train, ', test_samples:', l_test)                                        

    for epoch in range(num_epochs):
        train_loss, test_loss, start = 0.0, 0.0, time.time()        
        
        net.train()
        for _, data1 in enumerate(train_iter):
            X1, y1 = data1
            X1, y1 = X1.to(device), y1.to(device)                  
            output1 = net(X1)                                                   
            l1 = criterion(y1, output1)                              
            optimizer.zero_grad()                                  
            l1.backward()                                          
            optimizer.step()                                        
            train_loss += l1.cpu().item() * y1.size(0)             
        scheduler.step()

        net.eval()
        with torch.no_grad():
            for _, data2 in enumerate(test_iter):
                X2, y2 = data2
                X2 = X2.to(device)
                y2 = y2.to(device)                  
                output2 = net(X2)                
                l2 = criterion(y2, output2)                       
                test_loss += l2.cpu().item() * y2.size(0)

        train_log[epoch][0], train_log[epoch][1] = train_loss / l_train, test_loss / l_test  
        train_log[epoch][2], train_log[epoch][3] = 0, 0 
        train_log[epoch][4], train_log[epoch][5] = 0, 0    
        print('epoch {}, Train loss {:.6f}, Test loss: {:.6f}, time {:.1f}s'.format(epoch + 1, train_loss / l_train, test_loss / l_test, time.time()-start))
    
    torch.save(net, path1)                                                                           
    torch.save(net.state_dict(), path2)                                                               
    np.save('results/runs/' + str(dataset[0:2]) + '/MLP/'+ str(dataset) + '_' + str(model)+ '_' \
    + str(num_epochs) + 'epoch/train_log', train_log)                                               
    print('dataset = ' + str(dataset) + ', train_set samples:', l_train, ', test_samples:', l_test)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'vp_10_16ob'
    root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'
    train_path = '/mnt/share1/pengxingwen/Dataset/vp/train.txt'
    test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'
    batch_size = 256                                
    ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_c1_16.pt')

    model = 'MLP'
    num_inputs, num_outputs = 16, 52
    num_hiddens1, num_hiddens2 = 32, 32
    net = MLPV2(num_inputs, num_hiddens1, num_hiddens2, num_outputs)                                
    net = net.to(device)
                        
    criterion = torch.nn.L1Loss()       
    lr = 0.01                            
    num_epochs = 100                     
    st = time.time()                    
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)            
    scheduler = ExponentialLR(optimizer, gamma=0.97)                                    

    dirs = '/mnt/share1/pengxingwen/reconstruction_pxw/results/runs/' + str(dataset[0:2]) + '/MLP/'+ str(dataset) + '_' + str(model)+ '_' + str(num_epochs) + 'epoch'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    writer = SummaryWriter('/mnt/share1/pengxingwen/reconstruction_pxw/results/runs/' + str(dataset[0:2]) + '/MLP/'+ str(dataset) + '_' + str(model))
    path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset[0:2]) + '/MLP/' + str(dataset) + '_' + str(model) + '_' + str(num_epochs) + 'epoch.pt'
    path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset[0:2]) + '/MLP/' + str(dataset) + '_' + str(model) + '_' + str(num_epochs) + 'epoch_params.pt'
  
    print("trainning start")
    train(dataset, model, net, optimizer, device, num_epochs, criterion)
    print('model = ' + str(model) + ', loss = MAE, epoch = ' + str(num_epochs))
    train_log_draw(dataset, model, num_epochs) 
    print("trainning end, cost {:.4f}".format((time.time()-st) / 3600), 'h')
