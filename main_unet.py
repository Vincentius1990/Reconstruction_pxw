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
# from torch.utils.tensorboard import SummaryWriter
from src.data.dataset import MyNewData
from src.models.unet import UNetV2
# from src.models.unet_original import UNet
from src.loss.loss import GradientLoss, TVLoss, loss_error
from src.loss.train_log import train_log_draw


def train(dataset, model, net, optimizer, device, num_epochs, criterion, la):
    '''
    train the UNet
    '''

    train_dataset = MyNewData(root, train_path, ind, None)                                            
    train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle= True, num_workers=16)      
    test_dataset = MyNewData(root, test_path, ind, None)                                               
    test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)     
    l_train, l_test = train_dataset.__len__(), test_dataset.__len__()                                 
    train_log = np.zeros([num_epochs, 6])                                                             
    
    print('Training on', device, 'model = ' + str(model), 'loss = MAE+' + str(la)+ '*GradientLoss')           
    print('dataset = ' + str(dataset), 'train_samples:', l_train, ', test_samples:', l_test)                                 

    for epoch in range(num_epochs):
        train_loss, test_loss, start = 0.0, 0.0, time.time()       
        mae, cmae = 0.0, 0.0                                       
        MaxAE_batch, MTAE_batch = 0.0, 0.0                        
        MaxAE, MTAE = [], []                                       
        MaxAE_epoch, MTAE_epoch = 0.0, 0.0                         
        
        net.train()
        for _, data1 in enumerate(train_iter):
            X1, y1 = data1
            X1, y1 = X1.to(device), y1.to(device)                  
            output1 = net(X1)                                       
            
            lf1 = criterion(y1, output1)                           
            lg1 = gradient_loss(y1, output1)                                       
            l1 = lf1  + la * lg1                                     
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
                
                lf2 = criterion(y2, output2)                        
                lg2 = gradient_loss(y2, output2)                   
                lc2 = criterion(y2*cnd, output2*cnd)                       
                l2 = lf2  + la * lg2                                
                test_loss += l2.cpu().item() * y2.size(0)
                mae += lf2.cpu().item() * y2.size(0)
                cmae += lc2.cpu().item() * y2.size(0)
                
                MaxAE_batch, MTAE_batch = loss_error(output2, y2, batch_size)   
                MaxAE.append(MaxAE_batch.cpu().numpy().tolist())              
                MTAE.append(MTAE_batch.cpu().numpy().tolist())                 

        # 4 matrics: MAE, CMAE, MaxAE, MT-AE
        MaxAE, MTAE = torch.tensor(MaxAE).reshape(-1), torch.tensor(MTAE).reshape(-1)      
        MaxAE_epoch, MTAE_epoch = torch.max(MaxAE), torch.max(MTAE)                        E


        # train_loss, test_loss, mae, cmae, MaxAE_epoch, MTAE_epoch     
        train_log[epoch][0], train_log[epoch][1] = train_loss / l_train, test_loss / l_test
        train_log[epoch][2], train_log[epoch][3] = mae / l_test, cmae / l_test 
        train_log[epoch][4], train_log[epoch][5] = MaxAE_epoch, MTAE_epoch
                        
        print('epoch {}, Train loss {:.5f}, Test loss: {:.5f}, MAE: {:.5f}, CMAE: {:.5f}, MaxAE: {:.5f}, MT-AE: {:.5f},'
        ' time {:.1f}s'.format(epoch + 1, train_loss / l_train, test_loss / l_test, mae / l_test, cmae / l_test,  \
        MaxAE_epoch, MTAE_epoch,  time.time()-start))
    
    torch.save(net, path1)                                                                            
    torch.save(net.state_dict(), path2)                                                            
    np.savetxt('/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model)+ '/' \
    'train_log' + '_' + str(num_epochs) + 'epoch.txt', train_log)                                               
    print('dataset = ' + str(dataset) + ', train_samples:', l_train, ', test_samples:', l_test)                


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'lr0.005'
    root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'
    train_path = '/mnt/share1/pengxingwen/Dataset/vp/train.txt'
    test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'
    batch_size = 64                                 
    ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/ind_4.pt')
    cnd = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/ind/cnd_c1.pt')
    cnd = cnd.to(device) 

    model = 'UNetV2'          
    net = UNetV2(in_channels=1, num_classes=1)                      
    net = net.to(device)
                                 
    criterion = torch.nn.L1Loss()       
    gradient_loss = GradientLoss()                     
    lr = 0.005                         
    la = 0                              
    num_epochs = 30                   
    st = time.time()                 
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)            
    scheduler = ExponentialLR(optimizer, gamma=0.98)                 

    dirs = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model) + '/' + str(dataset) + '_' + str(model) + '_' + str(num_epochs) + 'epoch.pt'
    path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model) + '/' + str(dataset) + '_' + str(model) +'_' + str(num_epochs) + 'epoch_params.pt'
  
    print("Trainning start")
    train(dataset, model, net, optimizer, device, num_epochs, criterion, la)
    print('model = ' + str(model) + ', loss = MAE+' + str(la) + ' * GradientLoss, epoch = ' + str(num_epochs))
    train_log_draw(dataset, model, num_epochs) 
    print("trainning end, cost {:.4f}".format((time.time()-st) / 3600), 'hour')
