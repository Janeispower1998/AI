# 回归任务

# 步骤一 创建数据
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-1, 1, 100)  # 生成-1到1等差数列 一百个点 此时的数据是一维
# 训练集数据需要的是二维，float类型
# 升维
x = torch.unsqueeze(x, dim=1) # 此时数据为二维 (100，1)
print("x.shape,", x.numpy().shape)
# x的平方
y = x.pow(2) + 0.2 * (torch.rand(x.size()))

# plt.scatter(x.numpy(),y.numpy())
# plt.show()

# 创建模型 三层神经网络 输入层（一个神经元）隐藏层（十个神经元）输出层（一个神经元）
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h1 = nn.Linear(1, 8, bias=True)
        self.h2 = nn.Linear(8, 10)
        self.h3 = nn.Linear(10, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):  # 正向传播，定义神经网络层的顺序
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.h3(x)
        x = F.relu(x)
        o = self.out(x)
        return o

model = Net()
loss_fn = nn.MSELoss()

# 随机梯度下降（SGD）跟新梯度 创建对象
# model.parameters() 模型里所有的参数
optimizer=torch.optim.SGD(model.parameters(),\
          lr=0.3)

plt.ion() # 该行代码和最后两行代码用来显示动图
for i in range(100):
    o = model(x)  # o=model.forward(x)
    # 计算均方误差
    loss = loss_fn(o, y)  # （0-y）^2求和 tensor对象
    print(loss.item())  # 拿出tensor对象数据
    optimizer.zero_grad()  # 在梯度的存储空间中 把梯度全部清零
    # 反向传播跟新参数
    loss.backward()  # 求参数的偏导
    optimizer.step()
    if i % 5 == 0:
        plt.cla() # 清除轴，当前活动轴在当前图中。 它保持其他轴不变
        # 训练集数据
        plt.scatter(x.numpy(), y.numpy())
        # 预测数据
        plt.plot(x.numpy(), o.data.numpy(), 'r') # 'r'表示红色
        plt.pause(0.1)
plt.ioff()
plt.show()

# x = torch.randn(128, 20)  # 输入的维度是（128，20）
# m = torch.nn.Linear(20, 30)  # 20,30是指维度
# output = m(x)
# print('output.shape:\n', output.shape)



总结：
1.创建训练集数据
2.创建神经网络模型（包括损失函数、优化器）
3.通过调整参数找出最优function

神经网络同一套参数训练出来结果不一样：
1.权重参数的随机初始化。以一个四层神经网络为例，每层1024个神经元，那么一共需要初始化数百万个参数，显然是不能一一指定的
2.梯度下降算法一般都会使用随机梯度下降，或者mini batch，在执行每一次前向-单向传播中，用到的样本数据和顺序也是有一定随机性的
3.一些训练技巧本身也会引入随机性来对抗过拟合，比如 dropout。显然也没法指定每次dropout的时候都要舍弃哪些神经元
