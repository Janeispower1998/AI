#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    使用pytorch手撕神经网络
'''
#y = wx+b   y=wx
import torch
# #N有多少行数据，D_in神经网络输入层神经元个数，H隐藏层神经元个数，D_out输出层神经元个数
# N,D_in,H,D_out = 64,1000,100,10
# #创建训练集样本
# x = torch.randn(64,1000)
# y = torch.randn(64,10)  #目标值
# #随机初始化参数
# w1 = torch.randn(1000,100)
# w2 = torch.randn(100,10)
# lr = 1e-6
# for i in range(500):
# #步骤一：正向传播
#     print(i)
#     h  = x.mm(w1)  #mm矩阵乘法
#     # print(h)
#     print('-----------------------------------')
#     h_relu = h.clamp(min=0)  #激活函数，relu（）,小于0得部分都转化为0
#     # print(h_relu)
#     y_pred = h_relu.mm(w2)  #神经网络的预测结果
#     #步骤二：计算误差，均方误差
#     loss = (y_pred-y).pow(2).sum()  #(64,10)
#     print(loss.item())
#     #步骤三：反向传播
#     #w2的偏导
#     grad_y_pred = 2.0*(y_pred-y)
#     # print(h_relu.shape)
#     # print(grad_y_pred.shape)
#     grad_w2 = h_relu.t().mm(grad_y_pred)#结果，100*10#h_relu.T,(100,64)
#
#     #w1的偏导
#     # print(grad_y_pred.shape)
#     # print(w2.shape)
#     grad_y_relu = grad_y_pred.mm(w2.T) #(64,10)*(10,100) = (64,100)
#     grad_h = grad_y_relu.clone()
#     grad_h[h<0]=0
#     grad_w1 = x.t().mm(grad_h)
#     # print(grad_w1.shape)
#
#     #步骤四：更新参数
#     w1 -=lr*grad_w1
#     w2 -=lr*grad_w2


'''
    pytorch 可以自动求导

'''
# #Variable（变量），一定要是一个float tensor
# x = torch.tensor(1.,requires_grad=True)
# w = torch.tensor(2.,requires_grad = True)
# b = torch.tensor(3.,requires_grad = True)
#
#
# for i in range(10):
#     y = w * x + b   #正向传播
#     #print(y)
#     print(i)
#     print(x.grad)
#     y.backward()  #反向传播，在计算梯度
#     print(x.grad)  #变量的梯度值 2
#     print(w.grad) #1
#     print(b.grad)#1
#     x.grad.zero_()
#     w.grad.zero_()
#     b.grad.zero_()
#
#
#
# import torch
# #N有多少行数据，D_in神经网络输入层神经元个数，H隐藏层神经元个数，D_out输出层神经元个数
# N,D_in,H,D_out = 64,1000,100,10
# #创建训练集样本
# x = torch.randn(64,1000)
# y = torch.randn(64,10)  #目标值
# #随机初始化参数
# w1 = torch.randn(1000,100,requires_grad=True)
# w2 = torch.randn(100,10,requires_grad=True)
# lr = 1e-6
# for i in range(500):
# #步骤一：正向传播
#     print(i)
#     h  = x.mm(w1)  #mm矩阵乘法
#     # print(h)
#     print('-----------------------------------')
#     h_relu = h.clamp(min=0)  #激活函数，relu（）,小于0得部分都转化为0
#     # print(h_relu)
#     y_pred = h_relu.mm(w2)  #神经网络的预测结果
#     #步骤二：计算误差，均方误差
#     loss = (y_pred-y).pow(2).sum()  #(64,10)
#     print(loss.item())
#     #步骤三：反向传播
#     loss.backward()  #(固定的)
#     # print('w1的梯度：',w1.grad)
#     # print('w2的梯度：',w2.grad)
#
#     with torch.no_grad():  #Variable 不能用‘=’赋值
#     #步骤四：更新参数
#         w1 -=lr*w1.grad
#         w2 -=lr*w2.grad
#         w1.grad.zero_()
#         w2.grad.zero_()

'''
    快速搭建神经网络

'''

# import torch
# import torch.nn as nn  #创建神经网络使用的模块
#
# #N有多少行数据，D_in神经网络输入层神经元个数，H隐藏层神经元个数，D_out输出层神经元个数
# N,D_in,H,D_out = 64,1000,100,10
# #创建训练集样本
# x = torch.randn(64,1000)
# y = torch.randn(64,10)  #目标值
# # 此时不用定义权重,w1,w2,b1,b2
# #自定义神经网络模型
# model = nn.Sequential(
#     #y = wx+b
#     nn.Linear(1000,100,bias = False), #该层每个神经元有多少个输入，该层有多少个神经元，bias,是否加偏置
#     nn.ReLU(),
#     nn.Linear(100,10,bias = True)
# )
# #查看模型中的参数
# # print(model.parameters())
# # for param in model.parameters():
# #     print(param.size())
#
# #计算误差
# loss_fn = nn.MSELoss() #均方误差函数,一般在做回归问题的时候使用
# lr = 1e-6
# for i in range(500):
# #步骤一：正向传播
#     print(i)
#     y_pred = model(x)
#     #步骤二：计算误差，均方误差
#     loss = loss_fn(y_pred,y)
#     print(loss.item())
#     model.zero_grad()
#     #步骤三：反向传播
#     loss.backward()  #(固定的)
#     # print('w1的梯度：',w1.grad)
#     # print('w2的梯度：',w2.grad)
#     for param in model.parameters():
#         param -=lr*param.grad

#优化器，pytorch中更新参数的方式
# import torch
# import torch.nn as nn  #创建神经网络使用的模块
#
# #N有多少行数据，D_in神经网络输入层神经元个数，H隐藏层神经元个数，D_out输出层神经元个数
# N,D_in,H,D_out = 64,1000,100,10
# #创建训练集样本
# x = torch.randn(64,1000)
# y = torch.randn(64,10)  #目标值
# # 此时不用定义权重,w1,w2,b1,b2
# #自定义神经网络模型
# model = nn.Sequential(
#     #y = wx+b
#     nn.Linear(1000,100,bias = False), #该层每个神经元有多少个输入，该层有多少个神经元，bias,是否加偏置
#     nn.ReLU(),
#     nn.Linear(100,10,bias = True)
# )
# #查看模型中的参数
# # print(model.parameters())
# # for param in model.parameters():
# #     print(param.size())
#
# lr = 1e-7
# #计算误差
# loss_fn = nn.MSELoss() #均方误差函数,一般在做回归问题的时候使用
# optimizer = torch.optim.SGD(model.parameters(),lr = lr)
#
# for i in range(500):
# #步骤一：正向传播
#     print(i)
#     y_pred = model(x)
#     #步骤二：计算误差，均方误差
#     loss = loss_fn(y_pred,y)
#     print(loss.item())
#     optimizer.zero_grad()
#     #步骤三：反向传播
#     loss.backward()  #(固定的)
#     #步骤四：更新参数
#     optimizer.step()  #逐步更新神经网络中的参数
#



'''
    创建神经网络模型
'''

import torch
import torch.nn as nn  #创建神经网络各种功能层的模块
import torch.nn.functional as F  #在神经网络中计算会应用到的一些层

#N有多少行数据，D_in神经网络输入层神经元个数，H隐藏层神经元个数，D_out输出层神经元个数
N,D_in,H,D_out = 64,1000,100,10
#创建训练集样本
x = torch.randn(64,1000)
y = torch.randn(64,10)  #目标值
# 此时不用定义权重,w1,w2,b1,b2
#自定义神经网络模型
class ANN(nn.Module):
    def __init__(self,D_in,H,D_out):  #构造函数，定义所需要使用的神经网络层
        super(ANN,self).__init__()
        self.hidden = nn.Linear(D_in,H)
        #self.relu = nn.ReLU()
        self.out = nn.Linear(H,D_out)

    def forward(self,x):  #正向传播，定义神经网络层的顺序
        x = self.hidden(x)
        #x = self.relu(x)
        x  = F.relu(x)
        x = self.out(x)
        return x
model = ANN(1000,100,10)
#查看模型中的参数
# print(model.parameters())
# for param in model.parameters():
#     print(param.size())

lr = 1e-7
#计算误差
loss_fn = nn.MSELoss() #均方误差函数,一般在做回归问题的时候使用
optimizer = torch.optim.SGD(model.parameters(),lr = lr)

for i in range(500):
#步骤一：正向传播
    print(i)
    y_pred = model(x)
    #步骤二：计算误差，均方误差
    loss = loss_fn(y_pred,y)
    print(loss.item())
    optimizer.zero_grad()
    #步骤三：反向传播
    loss.backward()  #(固定的)
    #步骤四：更新参数
    optimizer.step()  #逐步更新神经网络中的参数








