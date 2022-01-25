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
    训练网络，记录模型参数
    '''

    # 读取训练集和测试集
    train_dataset = HeatsinkData(root, train_path, ind, None)                                             # 读取训练集
    train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle= True, num_workers=16)      # 封装训练集
    test_dataset = HeatsinkData(root, test_path, ind, None)                                               # 读取测试集
    test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)      # 封装测试集
    l_train, l_test = train_dataset.__len__(), test_dataset.__len__()                                  # 获取训练集和测试集样本数量
    train_log = np.zeros([num_epochs, 6])                                                              # 保存训练过程2个loss和4个指标
    
    print('Training on', device, 'model = ' + str(model), 'loss = MAE')             # 模型和loss信息
    print('dataset = ' + str(dataset), 'train_set samples:', l_train, ', test_samples:', l_test)       # 数据集信息                                         

    for epoch in range(num_epochs):
        ## 训练集上训练
        train_loss, test_loss, start = 0.0, 0.0, time.time()        # 训练误差 (MAELoss), 测试误差, 训练开始时间赋初值
        
        net.train()
        for _, data1 in enumerate(train_iter):
            X1, y1 = data1
            X1, y1 = X1.to(device), y1.to(device)                   # 数据转存到GPU上
            output1 = net(X1)                                       # 读取数据进行预测
            # print('X1:', X1.shape, 'y1:', y1.shape, 'output1:', output1.shape)             
            l1 = criterion(y1, output1)                            # L1 loss计算       
            optimizer.zero_grad()                                   # 梯度清零
            l1.backward()                                           # 反向传播
            optimizer.step()                                        # 参数优化
            train_loss += l1.cpu().item() * y1.size(0)              # train_loss累加
        scheduler.step()

        ## 测试集上测试
        net.eval()
        with torch.no_grad():
            for _, data2 in enumerate(test_iter):
                X2, y2 = data2
                X2 = X2.to(device)
                y2 = y2.to(device)                  
                output2 = net(X2)                
                l2 = criterion(y2, output2)                        # L1 loss 计算 
                test_loss += l2.cpu().item() * y2.size(0)

        train_log[epoch][0], train_log[epoch][1] = train_loss / l_train, test_loss / l_test  
        train_log[epoch][2], train_log[epoch][3] = 0, 0 
        train_log[epoch][4], train_log[epoch][5] = 0, 0    
        print('epoch {}, Train loss {:.6f}, Test loss: {:.6f}, time {:.1f}s'.format(epoch + 1, train_loss / l_train, test_loss / l_test, time.time()-start))
    
    torch.save(net, path1)                                                                           # 保存网络   
    torch.save(net.state_dict(), path2)                                                              # 保存模型参数  
    np.save('results/runs/' + str(dataset[0:2]) + '/MLP/'+ str(dataset) + '_' + str(model)+ '_' \
    + str(num_epochs) + 'epoch/train_log', train_log)                                                # 保存指标
    print('dataset = ' + str(dataset) + ', train_set samples:', l_train, ', test_samples:', l_test)


if __name__ == '__main__':
    '''
    主函数,指定路径,定义超参数,网络等
    '''

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
                        
    criterion = torch.nn.L1Loss()        # 训练loss, L1 loss, 乙可采用torch.nn.MSELoss(),效果相当
    lr = 0.01                            # 学习率
    num_epochs = 100                     # 训练迭代次数
    st = time.time()                     # 训练及测试总时间
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)            # 采用Adam优化算法
    scheduler = ExponentialLR(optimizer, gamma=0.97)                 # 指数变化学习率
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)         #采用Adam优化算法        
    # scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180], gamma=0.5)   #里程碑式变学习率
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)                        # SGD优化算法

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
