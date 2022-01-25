import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import sys

class MLP(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super(MLP,self).__init__()
        self.l1 = nn.Linear(num_inputs,num_hiddens1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(num_hiddens1,num_hiddens1)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(num_hiddens1,num_hiddens1)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(num_hiddens2,num_hiddens2)
        self.relu4 = nn.ReLU()
        self.l5 = nn.Linear(num_hiddens2,num_outputs)

    def forward(self,X):
        # X=X.view(X.shape[0],-1)        #二维展开成一维
        o1 = self.relu1(self.l1(X))
        o2 = self.l2(o1)
        o3 = self.relu2(o2)
        o4 = self.l3(o3)
        o5 = self.relu3(o4)
        o6 = self.l4(o5)
        o7 = self.relu4(o6)
        o8 = self.l5(o7)

        return o8

    def init_params(self):
        for param in self.parameters():
            print(param.shape)
            init.normal_(param,mean=0,std=0.01)


class MLPV2(nn.Module):
    def __init__(self, input_dim, hidden_num1, hidden_num2, output_dim):
        super(MLPV2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_num1),
            nn.GELU(),
            # nn.Linear(hidden_num, hidden_num),
            # nn.GELU(),
            # nn.Linear(hidden_num, hidden_num),
            # nn.GELU(),
            nn.Linear(hidden_num1, hidden_num2),
            nn.GELU(),
            nn.Linear(hidden_num2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 100, 400, 256, 256
    net = MLP(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    print(net)
    net.init_params()
    x = torch.randn(64, 1, 100)
    with torch.no_grad():
        y = net(x)
        # y = y.view(x.shape[0], 20, 20)
        print(y.shape)