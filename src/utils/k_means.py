import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
from  sklearn.cluster  import KMeans
from pylab import *

np.random.seed(0)
   

def calEuclideanDistance(vec1,vec2):
    '''
    计算两个向量之间的欧氏距离
    '''

    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist


def random_color(number):
    '''
    生成number种随机颜色
    '''

    color = []
    intnum = [str(x) for x in np.arange(10)]                 #['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet = [chr(x) for x in (np.arange(6) + ord('A'))]   #['A', 'B', 'C', 'D', 'E', 'F']
    colorArr = np.hstack((intnum, alphabet))                 #array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C','D', 'E', 'F'], dtype='<U1')
    for j in range(number):
        color_single = '#'
        for i in range(6):
            index = np.random.randint(len(colorArr))
            color_single += colorArr[index]                #随机生成一种'#81D4D4'式的颜色
        color.append(color_single)
    return color
    del color, intnum, alphabet, colorArr, j, i, color_single, index, number  #解除对变量的使用
  
def k_matrix():
    '''
    读取原始数据,#获取20k * 40k的二维矩阵
    '''

    u_data = loadmat('/mnt/share1/pengxingwen/Dataset/vp/vp_10_20000/Example0.mat')['u'].reshape(-1)
    u_data = np.expand_dims(u_data, 0).T
    # print('u_data :', type(u_data), u_data.shape)   #获取20k * 40k的二维矩阵
    # print(u_data)   
    for idx in range(1, 20000):
        u_data0 = loadmat('/mnt/share1/pengxingwen/Dataset/vp/vp_10_20000/Example'+str(idx)+'.mat')['u'].reshape(-1)
        u_data0 = np.expand_dims(u_data0, 0).T
        u_data = np.concatenate((u_data, u_data0), axis = 1)
        if idx % 10 == 0:
            print(u_data.shape)

    print('u_data :', type(u_data), u_data.shape)   #获取20k * 40k的二维矩阵
    # print(u_data)
    np.save('u_data', u_data)



def K_Means(n_clusters):
    '''
    读入矩阵，kmeans聚类分析
    '''

    u_data = np.load('u_data.npy')             # 采用全部样本
    X = pd.DataFrame(u_data)
    kmeans = KMeans(n_clusters)              # 构造聚类器
    kmeans.fit(X)                            # 聚类
    X['cluster'] = kmeans.labels_            # 获取聚类标签
    label_pred = kmeans.labels_              # 获取聚类标签
    label_pred = np.expand_dims(label_pred, axis=0)
    label_pred = label_pred.T
    print('Cluster results : ', type(label_pred), label_pred.shape)
    print(label_pred.T)
    X.cluster.value_counts()
    print('X with cluster label : ', type(X), X.shape)
    print(X)
    centers = kmeans.cluster_centers_         #获取聚类中心
    print('cluster centers: ', type(centers), centers.shape, '\n', centers)

    cluster_data = label_pred.reshape(200, 200)
    plt.imshow(cluster_data)      
    plt.title(str(n_clusters) + '-cluster_data')
    plt.savefig(str(n_clusters) + '-cluster_data.png')  

#————————————————————————————————计算每个点与各自聚类中心的距离———————————————————————————
    len_mp = 40000              
    dist = np.zeros((len_mp))
    X0 = np.array(X)                        
    print('X0 : ', type(X0), X0.shape)
    # print(X0)

    for i in range(len_mp):
        # print(X0[i,:-1], X0[i,-1])
        # print(centers[int(X0[i,-1])])
        dist[i] = calEuclideanDistance(X0[i,:-1], centers[int(X0[i,-1])])

    dist = np.expand_dims(dist, 0).T
    print('EuclideanDistance : ', type(dist), dist.shape)
    print(dist)

#————————————————————————————————输出聚类结果并按照欧式距离排序——————————————————————————————————————————
    order_data = np.arange(0,40000)
    # order_data = np.expand_dims(order_data, 0).T
    # print('order_data :', type(order_data), order_data.shape)
    # order_data = str(order_data)
    therm = np.column_stack((order_data, label_pred[:,0], dist))           #添加欧氏距离到聚类结果后面
    print('Order, Cluster, EuclideanDistance', therm.shape)
    print(therm)
    sortedindex = np.lexsort((therm[:, 1], therm[:, 2]))       #先按照聚类标签排序，再按照欧氏距离排序
    sort_therm = therm[sortedindex , :]
    print('Sorted Order, Cluster, EuclideanDistance: ', sort_therm.shape, '\n', sort_therm)

#————————————————————————————————根据欧氏距离选取测点——————————————————————————————————————————    
    all_points = np.arange(0, len_mp,1)
    selected_points_order = []
    flag = 0
    for i in range(len_mp):     
        if flag == sort_therm[i, 1]:
            selected_points_order.append(sort_therm[i,0])
            flag +=1 
    print('selected_points_order: ', len(selected_points_order), '\n', selected_points_order)
    print('all_points : ', len(all_points), '\n', all_points)  
    unselected_points = np.setdiff1d(all_points, selected_points_order)
    print('unselected_points : ', len(unselected_points), '\n', unselected_points) 

#————————————————————————————————绘图显示选取的点——————————————————————————————————————————
    observation_points = np.zeros((len_mp))
    for point in selected_points_order:
        observation_points[int(point)] = 1
    observation_points = observation_points.reshape(200, 200)
    np.save(str(n_clusters) + '-observation_points', observation_points)
    print(str(n_clusters) +'-observation_points: ', observation_points.shape)

    plt.imshow(observation_points)      
    plt.title(str(n_clusters) +'-selected points')
    plt.savefig(str(n_clusters) +'-selected points.png')  
    

K_Means(16)
