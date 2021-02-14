#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# #加载参数


class Net (torch.nn.Module):#5层神经网络
    def __init__(self,D_in,H,D_out):#构造函数
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(D_in,H,bias=True)  #F.relu()
        # self.hidden2 = nn.Linear(8,10)
        # self.hidden3 = nn.Linear(10,10)
        self.out = nn.Linear(H, D_out)

    def forward(self, x): #类中的函数
        x = self.hidden1(x)
        x = F.relu(x)
        # x = self.hidden2(x)
        # x = F.relu(x)
        # x = self.hidden3(x)
        # x = F.relu(x)
        o = self.out(x)
        return o
model = Net(1,10,1)
model.load_state_dict(torch.load('model_02.pkl'))  #只是load回来参数
o = model(torch.Tensor([[0.5]]))
print(o)
