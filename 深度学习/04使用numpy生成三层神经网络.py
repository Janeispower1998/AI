# 使用numpy生成三层神经网络，输入层1000个神经元，隐藏层（1）100个神经元，输出层10个神经元
# y = wx+b   y=wx

import numpy as np

# 创建训练集样本
x = np.random.randn(64, 1000)
y = np.random.randn(64, 10)
# 随机初始化参数
w1 = np.random.randn(1000, 100)
w2 = np.random.randn(100, 10)
# N有多少行数据，D_in神经网络输入层神经元个数，H隐藏层神经元个数，D_out输出层神经元个数
N, D_in, H, D_out = 64, 1000, 100, 10

lr = 1e-6  # 学习率

for i in range(500):
    # 步骤一：正向传播
    print(i)
    h = x.dot(w1)  # dot()矩阵乘法
    # print(h)
    print('-----------------------------------')
    h_relu = np.maximum(h, 0)
    # print(h_relu)
    y_pred = h_relu.dot(w2)  # 神经网络的预测结果
    # 步骤二：计算误差
    print('1', y_pred)
    print('2', y)
    print('y_pred-y', y_pred - y)
    print('np.square(y_pred-y)', np.square(y_pred - y))
    print('sum', np.square(y_pred - y).sum())
    loss = np.square(y_pred - y).sum()
    # 步骤三：反向传播
    # w2的偏导
    grad_y_pred = 2.0 * (y_pred - y)
    # print(h_relu.shape)
    # print(grad_y_pred.shape)
    grad_w2 = h_relu.T.dot(grad_y_pred)  # 结果，100*10#h_relu.T,(100,64)
    # w1的偏导
    # print(grad_y_pred.shape)
    # print(w2.shape)
    grad_y_relu = grad_y_pred.dot(w2.T)  # (64,10)*(10,100) = (64,100)
    grad_h = grad_y_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    # print(grad_w1.shape)
    # 步骤四：更新参数
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
