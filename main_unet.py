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
    训练网络，记录模型参数
    '''

    # 读取训练集和测试集
    train_dataset = MyNewData(root, train_path, ind, None)                                             # 读取训练集
    train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle= True, num_workers=16)      # 封装训练集
    test_dataset = MyNewData(root, test_path, ind, None)                                               # 读取测试集
    test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)      # 封装测试集
    l_train, l_test = train_dataset.__len__(), test_dataset.__len__()                                  # 获取训练集和测试集样本数量
    train_log = np.zeros([num_epochs, 6])                                                              # 保存训练过程2个loss和4个指标
    
    print('Training on', device, 'model = ' + str(model), 'loss = MAE+' + str(la)+ '*GradientLoss')             # 模型和loss信息
    print('dataset = ' + str(dataset), 'train_samples:', l_train, ', test_samples:', l_test)       # 数据集信息                                         

    for epoch in range(num_epochs):
        '''训练集上训练'''
        train_loss, test_loss, start = 0.0, 0.0, time.time()        # 训练误差 (MAELoss + Gradient Loss), 测试误差, 训练开始时间赋初值
        mae, cmae = 0.0, 0.0                                        # 整场平均绝对误差(MAE), 组件平均误差(CMAE)赋初值
        MaxAE_batch, MTAE_batch = 0.0, 0.0                          # 每一个batch的MaxAE和MT-AE赋初值
        MaxAE, MTAE = [], []                                        # 记录每一个epoch的MaxAE和MT-AE在列表中
        MaxAE_epoch, MTAE_epoch = 0.0, 0.0                          # 每一个epoch的MaxAE和MTAE赋初值
        
        net.train()
        for _, data1 in enumerate(train_iter):
            X1, y1 = data1
            X1, y1 = X1.to(device), y1.to(device)                   # 数据转存到GPU上
            output1 = net(X1)                                       # 读取数据进行预测
            # print('X1:', X1.shape, 'y1:', y1.shape, 'output1:', output1.shape) 
            
            lf1 = criterion(y1, output1)                            # L1 loss计算 
            lg1 = gradient_loss(y1, output1)                        # Gradient loss计算
            # lc1 = criterion(y1 * cnd, output1 * cnd)              # 组件loss, 用作评价指标, 训练过程无须计算       
            l1 = lf1  + la * lg1                                     # 总的loss = L1 loss + la * Gradient Loss
            optimizer.zero_grad()                                   # 梯度清零
            l1.backward()                                           # 反向传播
            optimizer.step()                                        # 参数优化
            train_loss += l1.cpu().item() * y1.size(0)              # train_loss累加
        scheduler.step()

        '''测试集上测试'''
        net.eval()
        with torch.no_grad():
            for _, data2 in enumerate(test_iter):
                X2, y2 = data2
                X2 = X2.to(device)
                y2 = y2.to(device)                  
                output2 = net(X2)
                
                lf2 = criterion(y2, output2)                        # L1 loss 计算 
                lg2 = gradient_loss(y2, output2)                    # Gradient loss 计算
                lc2 = criterion(y2*cnd, output2*cnd)                # 组件loss, 用作评价指标           
                l2 = lf2  + la * lg2                                 # 总的loss = L1 loss + la * Gradient Loss
                test_loss += l2.cpu().item() * y2.size(0)
                mae += lf2.cpu().item() * y2.size(0)
                cmae += lc2.cpu().item() * y2.size(0)
                
                MaxAE_batch, MTAE_batch = loss_error(output2, y2, batch_size)   # 调用函数获取每个batch的max error和max temperature error
                MaxAE.append(MaxAE_batch.cpu().numpy().tolist())                # 将每个batch的max error加入列表
                MTAE.append(MTAE_batch.cpu().numpy().tolist())                  # 将每个batch的max temperature error加入列表

        ## 4个评价指标：MAE, CMAE, MaxAE, MT-AE
        MaxAE, MTAE = torch.tensor(MaxAE).reshape(-1), torch.tensor(MTAE).reshape(-1)      # MaxAE和MTAE展开成一维
        MaxAE_epoch, MTAE_epoch = torch.max(MaxAE), torch.max(MTAE)                        # 全局最大误差MaxAE和全局最高温度误差MT-AE
        # print('MaxAE', MaxAE.shape, '\n', MaxAE, '\n', 'MT-AE', MTAE.shape, '\n', MTAE)
        # print('MaxAE_epoch', MaxAE_epoch.shape, '\n', MaxAE_epoch, '\n', 'MTAE_epoch', MTAE_epoch.shape, '\n', MTAE_epoch)

        # 记录训练测试故过程6个指标数据, train_loss, test_loss, mae, cmae, MaxAE_epoch, MTAE_epoch     
        train_log[epoch][0], train_log[epoch][1] = train_loss / l_train, test_loss / l_test
        train_log[epoch][2], train_log[epoch][3] = mae / l_test, cmae / l_test 
        train_log[epoch][4], train_log[epoch][5] = MaxAE_epoch, MTAE_epoch
        
        # Tensorboard记录训练测试故过程指标数据  
        # writer.add_scalar('Train_loss', train_loss / l_train, epoch + 1)
        # writer.add_scalar('Test_loss', test_loss / l_test, epoch + 1)
        # writer.add_scalar('MAE', mae / l_test, epoch + 1)
        # writer.add_scalar('CMAE', cmae / l_test, epoch + 1)
        # writer.add_scalar('MaxAE', MaxAE_epoch, epoch + 1)           
        # writer.add_scalar('MT-AE', MTAE_epoch, epoch + 1) 

        # 仅训练，最后测试一次
        # print('epoch {}, Train loss {:.5f}, time {:.1f}s'.format(epoch + 1, train_loss / l_train, time.time()-start))                  
        print('epoch {}, Train loss {:.5f}, Test loss: {:.5f}, MAE: {:.5f}, CMAE: {:.5f}, MaxAE: {:.5f}, MT-AE: {:.5f},'
        ' time {:.1f}s'.format(epoch + 1, train_loss / l_train, test_loss / l_test, mae / l_test, cmae / l_test,  \
        MaxAE_epoch, MTAE_epoch,  time.time()-start))
    
    torch.save(net, path1)                                                                           # 保存网络   
    torch.save(net.state_dict(), path2)                                                              # 保存模型参数  
    np.savetxt('/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model)+ '/' \
    'train_log' + '_' + str(num_epochs) + 'epoch.txt', train_log)                                                # 保存指标
    print('dataset = ' + str(dataset) + ', train_samples:', l_train, ', test_samples:', l_test)                  # 保存指标


if __name__ == '__main__':
    '''
    主函数,指定路径,定义超参数,网络等
    '''

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
    # net = UNet(n_channels=1, n_classes=1)            # 转置卷积UNet
    net = UNetV2(in_channels=1, num_classes=1)         # 双线性插值UNetV2                           
    net = net.to(device)
                                 
    criterion = torch.nn.L1Loss()        # 训练loss, L1 loss, 乙可采用torch.nn.MSELoss(),效果相当
    gradient_loss = GradientLoss()       # 训练loss, 梯度loss
    # tv_loss = TVLoss()                   # 训练loss, TV loss
    lr = 0.005                            # 学习率
    la = 0                               # 梯度loss权重
    num_epochs = 30                     # 训练迭代次数
    st = time.time()                     # 训练及测试总时间
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)            # 采用Adam优化算法
    scheduler = ExponentialLR(optimizer, gamma=0.98)                 # 指数变化学习率
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)         #采用Adam优化算法        
    # scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180], gamma=0.5)   #里程碑式变学习率
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)                        # SGD优化算法

    dirs = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    # writer = Su+3mmaryWriter('results/runs/' + str(dataset[0:2]) + '/'+ str(dataset) + '_' + str(model))
    path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model) + '/' + str(dataset) + '_' + str(model) + '_' + str(num_epochs) + 'epoch.pt'
    path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(dataset) + '_' + str(model) + '/' + str(dataset) + '_' + str(model) +'_' + str(num_epochs) + 'epoch_params.pt'
  
    print("Trainning start")
    train(dataset, model, net, optimizer, device, num_epochs, criterion, la)
    print('model = ' + str(model) + ', loss = MAE+' + str(la) + ' * GradientLoss, epoch = ' + str(num_epochs))
    train_log_draw(dataset, model, num_epochs) 
    print("trainning end, cost {:.4f}".format((time.time()-st) / 3600), 'hour')
