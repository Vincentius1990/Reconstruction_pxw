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

import torch
from torch.nn import MSELoss
from torch.nn.functional import conv2d, pad, interpolate
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradientLoss(_Loss):

    def __init__(self):
        super().__init__()

    def forward(self, y, y_predict):
        batch_size = y.size()[0]  
        h_x = y.size()[2]        
        w_x = y.size()[3]         

        count_h = self._tensor_size(y[:,:,1:,:])     
        count_w = self._tensor_size(y[:,:,:,1:])
        # print(count_h, count_w)

        h1_tv = torch.abs(y[:,:,1:,:] - y[:,:,:h_x-1,:])                    
        h2_tv = torch.abs(y_predict[:,:,1:,:] - y_predict[:,:,:h_x-1,:])   
        h_gt= torch.abs(h1_tv - h2_tv).sum()
        w1_tv = torch.abs(y[:,:,:,1:] - y[:,:,:,:w_x-1])                    
        w2_tv = torch.abs(y_predict[:,:,:,1:] - y_predict[:,:,:,:w_x-1])    
        w_gt= torch.abs(w1_tv - w2_tv).sum()
        return (h_gt / count_h + w_gt / count_w) / batch_size
 
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def loss_error(y_pred, y, batch_size): 

    #[batch, 40000]
    y_pred = y_pred.reshape(batch_size, -1)  
    y = y.reshape(batch_size, -1)  
    # print(y_pred.shape, y.shape)  

    # batch MaxAE
    MaxAE, _ = torch.max(y_pred - y, 1)
    MaxAE = torch.abs(MaxAE)
    # print('MaxAE', MaxAE.shape, '\n', MaxAE)

    # batch MT-AE
    y_pred_max, _ = torch.max(y_pred, 1)      
    y_max, _ = torch.max(y, 1)                
    MTAE = torch.abs(y_pred_max - y_max)
    # print('MT-AE', MTAE.shape, '\n', MTAE)
    return MaxAE, MTAE  
