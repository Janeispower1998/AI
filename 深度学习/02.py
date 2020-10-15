# 回归任务
# 步骤一 创建数据
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#生成-1到1等差数列 一百个点
x=torch.linspace(-1,1,100)# 训练集数据需要的是二维，float类型
# 升维
x=torch.unsqueeze(x,dim=1)
print("x.shape,",x.numpy().shape)
#print(x)
# x的平方
y=x.pow(2)+0.2*(torch.rand(x.size()))
# plt.scatter(x.numpy(),y.numpy())
# plt.show()

# 创建模型 三层神经网络 输入层（一个神经元）
#                     ,隐藏层（十个神经元）
#                      输出层（一个神经元）

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.h1 = nn.Linear(1,8,bias=True)
        self.h2 = nn.Linear(8,10)
        self.h3 = nn.Linear(10,10)
        self.out = nn.Linear(10,1)

    def forward(self, x):  # 正向传播，定义神经网络层的顺序
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.h3(x)
        x = F.relu(x)
        o = self.out(x)
        return o
model=Net()
loss_fn=nn.MSELoss()
#随机梯度下降（SGD）跟新梯度 创建对象
#model.parameters() 模型里所有的参数
optimizer=torch.optim.SGD(model.parameters(),\
          lr=0.3)
plt.ion()
for i in range(100):
    o=model(x)#o=model.forward(x)
    #计算均方误差
    loss=loss_fn(o,y)# （0-y）^2求和 tensor对象
    print(loss.item())#拿出tensor对象数据
    optimizer.zero_grad()#在梯度的存储空间中 把梯度全部清零
    #反向传播跟新参数
    loss.backward()#求参数的偏导
    optimizer.step()
    if i%5==0:
        plt.cla()
        #训练集数据
        plt.scatter(x.numpy(),y.numpy())
        #预测数据
        plt.plot(x.numpy(),o.data.numpy(),'r')
        plt.pause(0.1)
plt.ioff()
plt.show()





# x = torch.randn(128, 20)  # 输入的维度是（128，20）
# m = torch.nn.Linear(20, 30)  # 20,30是指维度
# output = m(x)
# print('output.shape:\n', output.shape)