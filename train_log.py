import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def train_log_draw(dataset, model, num_epochs):
    '''
    绘制训练曲线
    '''
    if str(model) == 'MLP':
        train_log = np.load('/mnt/share1/pengxingwen/reconstruction_pxw/results/runs/' \
    + str(dataset[0:2]) + '/MLP/'+ str(dataset) + '_' + str(model)+ '_' + str(num_epochs) + 'epoch/train_log.npy')
        path = 'results/runs/' + str(dataset[0:2]) + '/MLP/'+ str(dataset) + '_'\
     + str(model)+ '_' + str(num_epochs) + 'epoch/Train_curve.png'
    else:
        train_log = np.load('/mnt/share1/pengxingwen/reconstruction_pxw/results/runs/' \
    + str(dataset[0:2]) + '/'+ str(dataset) + '_' + str(model)+ '_' + str(num_epochs) + 'epoch/train_log.npy')
        path = 'results/runs/' + str(dataset[0:2]) + '/'+ str(dataset) + '_'\
     + str(model)+ '_' + str(num_epochs) + 'epoch/Train_curve.png'
    
    print('train_log: \nTrain loss = {:.6f} \nTest loss = {:.6f} \nMAE = {:.6f} \nCMAE = {:.6f}K' 
    '\nMaxAE = {:.6f}K \nMT-AE = {:.6f}K '.format(train_log[-1][0], \
    train_log[-1][1], train_log[-1][2], train_log[-1][3], train_log[-1][4], train_log[-1][5]))

    x_epoch = range(num_epochs-1)
    fig = plt.figure(figsize=(20, 24))
    plt.subplot(3, 1, 1)  
    plt.title('Train Curve: Train loss = {:.4f}, Test loss = {:.4f}'\
    .format(train_log[-1][0], train_log[-1][1]))   
    plt.plot(x_epoch, train_log[1:,0], 'r', label=u'Train loss')
    plt.plot(x_epoch, train_log[1:,1], 'b', label=u'Test loss')
    plt.xlabel("Epochs")                              # X轴标签
    plt.ylabel("Loss")                                # Y轴标签
    plt.legend()                                      # 让图例生效

    plt.subplot(3, 1, 2) 
    plt.title('Error Curve: MAE = {:.4f}, CMAE = {:.4f}'\
    .format(train_log[-1][2], train_log[-1][3]))
    plt.plot(x_epoch, train_log[1:,2], 'g', label=u'MAE')
    plt.plot(x_epoch, train_log[1:,3], 'c', label=u'CMAE') 
    plt.xlabel("Epochs")                              # X轴标签
    plt.ylabel("Error")                               # Y轴标签
    plt.legend()                                      # 让图例生效

    plt.subplot(3, 1, 3) 
    plt.title('Error Curve: MaxAE = {:.4f}, MT-AE = {:.4f}'\
    .format(train_log[-1][4], train_log[-1][5]))
    plt.plot(x_epoch, train_log[1:,4], 'g', label=u'MaxAE') 
    plt.plot(x_epoch, train_log[1:,5], 'g', label=u'MT-AE') 
    plt.xlabel("Epochs")                              # X轴标签
    plt.ylabel("Error")                               # Y轴标签
    plt.legend()                                      # 让图例生效

    plt.savefig(path, bbox_inches='tight', pad_inches=0.3) 


# dataset = 'vp_10_c3_4ob'
# model = 'UNetV2'
# num_epochs = 100
# train_log_draw(dataset, model, num_epochs)

