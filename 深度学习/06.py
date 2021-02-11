import torch
#N有多少行数据，D_in神经网络输入层神经元个数，H隐藏层神经元个数，D_out输出层神经元个数
N,D_in,H,D_out = 64,1000,100,10
#创建训练集样本
x = torch.randn(64,1000)
y = torch.randn(64,10)  #目标值
#随机初始化参数
w1 = torch.randn(1000,100,requires_grad=True)
w2 = torch.randn(100,10,requires_grad=True)

lr = 1e-6

for i in range(500):
#步骤一：正向传播
    print(i)
    h  = x.mm(w1)  #mm矩阵乘法
    # print(h)
    print('-----------------------------------')
    h_relu = h.clamp(min=0)  #激活函数，relu（）,小于0得部分都转化为0
    # print(h_relu)
    y_pred = h_relu.mm(w2)  #神经网络的预测结果
    #步骤二：计算误差，均方误差
    loss = (y_pred-y).pow(2).sum()  #(64,10)
    print(loss.item())
    #步骤三：反向传播
    loss.backward()  #(固定的)
    # print('w1的梯度：',w1.grad)
    # print('w2的梯度：',w2.grad)

    with torch.no_grad():  #Variable 不能用‘=’赋值
    #步骤四：更新参数
        w1 -=lr*w1.grad
        w2 -=lr*w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
