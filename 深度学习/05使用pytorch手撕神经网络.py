# 使用pytorch手撕神经网络
# y = wx+b   y=wx

import torch

# N有多少行数据，D_in神经网络输入层神经元个数，H隐藏层神经元个数，D_out输出层神经元个数
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建训练集样本
x = torch.randn(64, 1000)
y = torch.randn(64, 10)  # 目标值
# 随机初始化参数
w1 = torch.randn(1000, 100)
w2 = torch.randn(100, 10)

lr = 1e-6

for i in range(500):
    # 步骤一：正向传播
    print(i)
    h = x.mm(w1)  # mm矩阵乘法
    # print(h)
    print('-----------------------------------')
    h_relu = h.clamp(min=0)  # 激活函数，relu（）,小于0得部分都转化为0
    # print(h_relu)
    y_pred = h_relu.mm(w2)  # 神经网络的预测结果
    # 步骤二：计算误差，均方误差
    loss = (y_pred - y).pow(2).sum()  # (64,10)
    print(loss.item())
    # 步骤三：反向传播
    # w2的偏导
    grad_y_pred = 2.0 * (y_pred - y)
    # print(h_relu.shape)
    # print(grad_y_pred.shape)
    grad_w2 = h_relu.t().mm(grad_y_pred)  # 结果，100*10#h_relu.T,(100,64)
    # w1的偏导
    # print(grad_y_pred.shape)
    # print(w2.shape)
    grad_y_relu = grad_y_pred.mm(w2.T)  # (64,10)*(10,100) = (64,100)
    grad_h = grad_y_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    # print(grad_w1.shape)
    # 步骤四：更新参数
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
