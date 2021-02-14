import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 步骤一：创建数据
x = torch.linspace(-1, 1, 100)  # 训练集数据需要是二维，float类型
x = torch.unsqueeze(x, dim=1)
print(x.shape)
print(x.dtype)
print(x)
y = x.pow(2) + 0.2 * (torch.rand(x.size()))


# plt.scatter(x.numpy(),y.numpy())
# plt.show()

# 创建模型，3层神经网络，输入层（1个神经元），隐藏层（8个神经元），输出层（1）
class Net(torch.nn.Module):  # 5层神经网络
    def __init__(self, D_in, H, D_out):  # 构造函数
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(D_in, H, bias=True)  # F.relu()
        # self.hidden2 = nn.Linear(8,10)
        # self.hidden3 = nn.Linear(10,10)
        self.out = nn.Linear(H, D_out)

    def forward(self, x):  # 类中的函数
        x = self.hidden1(x)
        x = F.relu(x)
        # x = self.hidden2(x)
        # x = F.relu(x)
        # x = self.hidden3(x)
        # x = F.relu(x)
        o = self.out(x)
        return o


model = Net(1, 10, 1)
# model2 = Net(10,20,2)
loss_fn = nn.MSELoss()
# 优化，参数初始化，lr（鲸鱼算法、蝙蝠算法、蜻蜓算法等），模型结构
# 三层神经网络，loss  =0.009889073669910431
# lr = 0.3 ,loss = 0.007699563633650541,,0.26
# lr = 0.02   loss = 0.0788702592253685
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
plt.ion()
# 训练模型
for i in range(100):
    o = model.forward(x)  # 在调用forward函数
    # 对比真实值和预测值，均方误差
    loss = loss_fn(o, y)  # (o-y)^2
    print(loss.item())
    optimizer.zero_grad()  # 在梯度的存储空间中，把梯度全部清零
    # 反向传播，更新参数
    loss.backward()  # 在求参数的偏导
    optimizer.step()
    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.numpy(), y.numpy())  # 训练集数据
        plt.plot(x.numpy(), o.data.numpy(), 'r')
        plt.pause(0.1)
plt.ioff()
plt.show()

# 保存参数
# 方式一：既保存模型，也保存参数
torch.save(model, 'model_01.pkl')
# 方式二：只保存参数（先定义模型，再定义参数）
# model.state_dict() model参数 字典类型
torch.save(model.state_dict(), 'model_02.pkl')

# 加载参数
model = torch.load('model_01.pkl')
o = model(torch.Tensor([[0.5], [0.3]]))
print("预测:{}".format(o))
