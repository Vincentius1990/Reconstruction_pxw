import time
import torch
from torch.utils.data import DataLoader
import src.loss as loss
from src.data.dataset import MyNewData
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(logname1, logname2, datasetname):
    '''
    测试模型效果
    '''
    st = time.time()   #  测试开始时间
    print('Loss = MAE \nmodel = ' + str(logname1) + '_' + str(logname2) + ' \ndataset = ' + str(datasetname), l_test)

    # 读取UNet模型
    tfr_path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(logname1[0:2]) + '/' + str(logname1) + '.pt'
    tfr_path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(logname1[0:2]) + '/' + str(logname1) + '_params.pt'
    net = torch.load(tfr_path1)                           #载入模型
    net.load_state_dict(torch.load(tfr_path2))            #载入参数

    if logname2 != None:
        # 读取MLP模型
        mlp_path1 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(logname2[0:2]) + '/MLP/' + str(logname2) + '.pt'
        mlp_path2 = '/mnt/share1/pengxingwen/reconstruction_pxw/results/' + str(logname2[0:2]) + '/MLP/' + str(logname2) + '_params.pt'
        mlp = torch.load(mlp_path1)                           #载入模型
        mlp.load_state_dict(torch.load(mlp_path2))            #载入参数
       
    # 评价指标
    MaxAE_epoch, MTAE_epoch = 0.0, 0.0
    MaxAE, MTAE = [], []                                   # 记录每一个epoch的 MaxAE, MT-AE
    MAE, CMAE = 0.0, 0.0                                   # 记录测试误差 MAE, 组件平均误差CMAE 
    MaxAE_batch, MTAE_batch = [], []                       # 记录每一个batch的 MaxAE, MT-AE
    criterion = torch.nn.L1Loss()                          # 训练loss, L1 loss, 亦可采用torch.nn.MSELoss(),效果相当
             

    net.eval()
    with torch.no_grad():
        for _, data in enumerate(test_iter):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            output = net(X)                                              # UNet预测温度场

            if logname2 != None:
                observation_y = X[X > 0]                                 # 提取温度测点数据
                observation_y = (observation_y - 298) / 50               # 归一化
                # print(i, observation_y.shape)
                observation_y = observation_y.reshape(-1, num_input)            # MLP输入                                
                heatsink_y = mlp(observation_y)                          # MLP预测温度场
                heatsink_y = (heatsink_y * 50) + 298                     # 归一化后还原
                heatsink_y = heatsink_y.reshape(-1, 1, 2, 26)            # MLP 输出

                output[:, :, 0:2, 86:112] = heatsink_y                   # 热沉区域替换
           
            l2 = criterion(y, output)                                    # MAE loss, 第一评价指标
            lc2 = criterion(y * cnd, output * cnd)                       # 组件loss，第二评价指标 
            MAE += l2.cpu().item() * y.size(0)
            CMAE += lc2.cpu().item() * y.size(0)
            
            MaxAE_batch, MTAE_batch = loss.loss_error(output, y, batch_size)       # 调用函数获取每个batch的MaxAE和MTAE
            MaxAE.append(MaxAE_batch.cpu().numpy().tolist())                       # 将每个batch的MaxAE加入列表
            MTAE.append(MTAE_batch.cpu().numpy().tolist())                         # 将每个batch的MTAE加入列表

        MaxAE = torch.tensor(MaxAE).reshape(-1)
        MTAE = torch.tensor(MTAE).reshape(-1)
        # print('MaxAE', MaxAE.shape, '\n', MaxAE, 'MTAE', MTAE.shape, '\n', MTAE)

        # MaxAE_epoch = torch.max(MaxAE)                                      # 全局最大误差 
        # MTAE_epoch = torch.max(MTAE)                                        # 全局最高温度误差
        MaxAE_epoch = torch.mean(MaxAE)                                      # 平均最大误差 
        MTAE_epoch = torch.mean(MTAE)                                        # 平均最高温度误差

        print('MAE: {:.6f} \nCMAE: {:.6f} \nMaxAE: {:.6f} \nMT-AE: {:.6f} \ntime {:.1f}s' \
        .format(MAE / l_test, CMAE / l_test, MaxAE_epoch, MTAE_epoch, time.time() - st))


if __name__ == '__main__':
    '''
    主函数,读取测试集,测试不同模型的效果
    '''

    '''--------------------Case 1---------------------'''
    cnd = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/cnd_c1.pt')
    cnd = cnd.to(device)   

    '''第一大类，general数据集测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_4.pt')
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'  
    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()                               
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16      
    # test('vp_10_c1_0.1grad_4ob_UNetV2_200epoch', None, 'test.txt')
    # test('vp_10_c1_0.1grad_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'test.txt')

    '''第一大类，special数据集测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_4.pt')
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_sp'
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/sp_1024.txt' 
    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()                               
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16         
    # test('vp_10_c1_0.1grad_4ob_UNetV2_200epoch', None, 'sp_1024.txt')
    # test('vp_10_c1_0.1grad_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'sp_1024.txt')  

    # test('vp_10_c1_4ob_UNetV2_200epoch', None, 'sp_1024.txt')
    # test('vp_10_c1_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'sp_1024.txt')

    '''第二大类，数据集规模测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_4.pt')
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/test_list_5k.txt'   
    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()                               
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16  
    # test('vp_10_c1_ts_40k_4ob_UNetV2_200epoch', None, 'test_list_5k.txt')
    # test('vp_10_c1_ts_40k_4ob_UNetV2_200epoch', 'vp_10_c1_4ob_MLP_100epoch', 'test_list_5k.txt')

    '''第三大类，观测点数量测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_1.pt')   # 观测点遮罩
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'                              # 原始数据
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'                          # 测试集

    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()                                
    # ind = ind.reshape(1, 200, 200)
    # num_input = 1                                                                     # MLP输入

    # test('vp_10_c1_1ob_UNetV2_200epoch', None, 'test.txt')
    # test('vp_10_c1_1ob_UNetV2_200epoch', 'vp_10_c1_1ob_MLP_100epoch', 'test.txt')
    
    '''第四大类，不同测点采样策略测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_c1_16.pt')   # 观测点遮罩
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16    
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c1_60k'                              # 原始数据
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'                          # 测试集

    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()  
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16                                

    # test('vp_10_c1_16ob_UNetV2_200epoch_2', None, 'test.txt')
    # test('vp_10_c1_16ob_UNetV2_200epoch_2', 'vp_10_c1_16ob_MLP_100epoch', 'test.txt')


    '''--------------------Case 3---------------------'''
    cnd = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/cnd_c3.pt')
    cnd = cnd.to(device)   
 
    '''第一大类，general数据集测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_4.pt')
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c3_55k'
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'      # test_list_5k.txt 
    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()                               
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16                               

    # test('vp_c3_0.1grad_4ob_UNetV2_200epoch', None, 'test.txt')
    # test('vp_c3_0.1grad_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'test.txt')
    # test('vp_c3_4ob_UNetV2_200epoch_4', None, 'test.txt')
    # test('vp_c3_4ob_UNetV2_200epoch_4', 'vp_c3_4ob_MLP_100epoch', 'test.txt')

    '''第一大类，special数据集测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_4.pt')
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c3_sp'
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/sp_1024.txt' 
    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()                               
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16  
    # test('vp_c3_4ob_UNetV2_200epoch_4', None, 'sp_1024.txt')
    # test('vp_c3_4ob_UNetV2_200epoch_4', 'vp_c3_4ob_MLP_100epoch', 'test.txt')

    '''第二大类，数据集规模测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_4.pt')
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c3_55k'
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/c3_test_list_5k.txt'   
    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()                               
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16  
    # test('vp_c3_ts_40k_4ob_UNetV2_200epoch', None, 'c3_test_list_5k.txt')
    # test('vp_c3_ts_40k_4ob_UNetV2_200epoch', 'vp_c3_4ob_MLP_100epoch', 'c3_test_list_5k.txt')

    '''第三大类，观测点数量测试'''
    ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_1.pt')   # 观测点遮罩
    root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c3_55k'                              # 原始数据
    test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'                          # 测试集

    test_dataset = MyNewData(root, test_path, ind, None)
    batch_size = 16 
    test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    l_test = test_dataset.__len__()                                
    ind = ind.reshape(1, 200, 200)
    num_input = 1                                                                     # MLP输入

    test('vp_c3_1ob_UNetV2_200epoch', None, 'test.txt')
    test('vp_c3_1ob_UNetV2_200epoch', 'vp_c3_1ob_MLP_100epoch', 'test.txt')

    '''第四大类，不同测点采样策略测试'''
    # ind = torch.load('/mnt/share1/pengxingwen/reconstruction_pxw/src/data/ind_c3_16.pt')   # 观测点遮罩
    # ind = ind.reshape(1, 200, 200)
    # num_input = 16    
    # root = '/mnt/share1/pengxingwen/Dataset/vp/vp_c3_55k'                              # 原始数据
    # test_path = '/mnt/share1/pengxingwen/Dataset/vp/test.txt'                          # 测试集

    # test_dataset = MyNewData(root, test_path, ind, None)
    # batch_size = 16 
    # test_iter  = DataLoader(test_dataset, batch_size = batch_size, shuffle= False, num_workers=16)
    # l_test = test_dataset.__len__()                                
                                                                

    # test('vp_c3_16ob_UNetV2_200epoch', None, 'test.txt')
    # test('vp_c3_16ob_UNetV2_200epoch', 'vp_c3_16ob_MLP_100epoch', 'test.txt')
