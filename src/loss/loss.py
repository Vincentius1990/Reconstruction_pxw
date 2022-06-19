import torch
from torch.nn import MSELoss
from torch.nn.functional import conv2d, pad, interpolate
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class PointLoss(_Loss):
    '''
    计算监测点处误差
    '''

    def __init__(self):
        super().__init__()

    def forward(self, monitor_y, layout_pred, criterion = torch.nn.L1Loss()):
        ones = torch.ones_like(monitor_y).cuda()
        zeros = torch.zeros_like(monitor_y).cuda()
        ind = torch.where(monitor_y> 0, ones, zeros)
        return criterion(monitor_y, layout_pred * ind)


class ComponentLoss(_Loss):
    '''
    计算组件上的平均MAE误差
    '''

    def __init__(self):
        super().__init__()

    def forward(self, u, layout_pred, cnd, criterion = torch.nn.L1Loss()):
        return criterion(u * cnd, layout_pred * cnd)


class GradientLoss(_Loss):
    '''
    计算梯度loss
    x[:, :, 1:, :] - x[:, :, :h_x-1, :] 就是对原图进行错位，分成两张像素位置差1的图片，
    第一张图片从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个像素点，            
    这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相邻的下一个像素点的差。
    '''

    def __init__(self):
        super().__init__()

    def forward(self, y, y_predict):
        batch_size = y.size()[0]  # batch size 批次大小
        h_x = y.size()[2]         # 图像的长
        w_x = y.size()[3]         # 图像的宽
        # 算出总共求了多少次差
        count_h = self._tensor_size(y[:,:,1:,:])     
        count_w = self._tensor_size(y[:,:,:,1:])
        # print(count_h, count_w)
        # 两个方向上求两张图的差
        h1_tv = torch.abs(y[:,:,1:,:] - y[:,:,:h_x-1,:])                    # h方向的梯度
        h2_tv = torch.abs(y_predict[:,:,1:,:] - y_predict[:,:,:h_x-1,:])    # h方向的梯度
        h_gt= torch.abs(h1_tv - h2_tv).sum()
        w1_tv = torch.abs(y[:,:,:,1:] - y[:,:,:,:w_x-1])                    # w方向的梯度
        w2_tv = torch.abs(y_predict[:,:,:,1:] - y_predict[:,:,:,:w_x-1])    # w方向的梯度
        w_gt= torch.abs(w1_tv - w2_tv).sum()
        return (h_gt / count_h + w_gt / count_w) / batch_size
 
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLoss(_Loss):
    '''
    计算TV loss
    '''

    def __init__(self):
        super().__init__()

    def forward(self, layout_pred):
        batch_size = layout_pred.size()[0]
        h_x = layout_pred.size()[2]
        w_x = layout_pred.size()[3]
        count_h = self._tensor_size(layout_pred[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(layout_pred[:,:,:,1:])
        h_tv = torch.pow((layout_pred[:,:,1:,:] - layout_pred[:,:,:h_x-1,:]),2).sum()  
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个            
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((layout_pred[:,:,:,1:] - layout_pred[:,:,:,:w_x-1]), 2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class LaplaceLoss(_Loss):
    '''
    计算物理loss
    '''

    def __init__(
        self, base_loss=MSELoss(reduction='mean'), nx=200,
        length=0.1, weight=[[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], bcs=[[[0.0495, 0], [0.0505, 0]]],
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weight = torch.Tensor(weight)
        self.bcs = bcs
        self.length = length
        self.nx = nx
        self.scale_factor = 1                       # self.nx/200
        TEMPER_COEFFICIENT = 50.0
        STRIDE = self.length / self.nx
        self.cof = -1 * STRIDE**2/TEMPER_COEFFICIENT

    def laplace(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat):
        N, C, W, H = layout.shape
        layout = interpolate(layout, scale_factor=self.scale_factor)
        heat = pad(heat, [1, 1, 1, 1], mode='replicate')    # constant, reflect, replicate
        layout_pred = self.laplace(heat)
        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            return self.base_loss(layout_pred[..., 1:-1, 1:-1], self.cof * layout[..., 1:-1, 1:-1])
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    layout_pred[..., idx_start:idx_end, :1] = self.cof * layout[..., idx_start:idx_end, :1]
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    layout_pred[..., idx_start:idx_end, -1:] = self.cof * layout[..., idx_start:idx_end, -1:]
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    layout_pred[..., :1, idx_start:idx_end] = self.cof * layout[..., :1, idx_start:idx_end]
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    layout_pred[..., -1:, idx_start:idx_end] = self.cof * layout[..., -1:, idx_start:idx_end]
                else:
                    raise ValueError("bc error!")
            return self.base_loss(layout_pred, self.cof * layout)


class OutsideLoss(_Loss):
    '''
    计算边界loss
    '''

    def __init__(
        self, base_loss=MSELoss(reduction='mean'), length=0.1, u_D=298, bcs=[[[0.0495, 0], [0.0505, 0]]], nx=200
    ):
        super().__init__()
        self.base_loss = base_loss
        self.u_D = u_D
        # bcs:
        #     - [[0.01, 0], [0.02, 0]]
        #     - [[0.08, 0], [0.09, 0]] # 2d example
        self.slice_bcs = []
        self.bcs = bcs
        self.nx = nx
        self.length = length

    def forward(self, x):
        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all bcs are Dirichlet
            d1 = x[:, :, :1, :]
            d2 = x[:, :, -1:, :]
            d3 = x[:, :, 1:-1, :1]
            d4 = x[:, :, 1:-1, -1:]
            point = torch.cat([d1.flatten(), d2.flatten(), d3.flatten(), d4.flatten()], dim=0)
            return self.base_loss(point, torch.ones_like(point) * 0)
        loss = 0
        for bc in self.bcs:
            if bc[0][1] == 0 and bc[1][1] == 0:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = x[..., idx_start:idx_end, :1]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][1] == self.length and bc[1][1] == self.length:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = x[..., idx_start:idx_end, -1:]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][0] == 0 and bc[1][0] == 0:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = x[..., :1, idx_start:idx_end]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][0] == self.length and bc[1][0] == self.length:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = x[..., -1:, idx_start:idx_end]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            else:
                raise ValueError("bc error!")
        return loss


def loss_error(y_pred, y, batch_size): 
    '''
    计算每个batch的MaxAE和MT-AE
    ''' 

    #调整批次的维度，变成[batch, 40000]
    y_pred = y_pred.reshape(batch_size, -1)  
    y = y.reshape(batch_size, -1)  
    # print(y_pred.shape, y.shape)  

    #计算一个batch的MaxAE
    MaxAE, _ = torch.max(y_pred - y, 1)
    MaxAE = torch.abs(MaxAE)
    # print('MaxAE', MaxAE.shape, '\n', MaxAE)

    #计算一个batch的MT-AE
    y_pred_max, _ = torch.max(y_pred, 1)      #按维度找出最大值
    y_max, _ = torch.max(y, 1)                #按维度找出最大值
    MTAE = torch.abs(y_pred_max - y_max)
    # print('MT-AE', MTAE.shape, '\n', MTAE)
    return MaxAE, MTAE  
