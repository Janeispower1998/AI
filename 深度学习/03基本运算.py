import torch

a = torch.Tensor([[1, 2], [2, 3], [3, 4]])  # 默认是浮点类型
print(a)
print(type(a))  # <class 'torch.Tensor'>
print(a.dtype)  # torch.float32

a = a.int()
print(a.dtype)  # torch.int32
a = a.long()
print(a.dtype)  # torch.int64
a = a.double()
print(a.dtype)  # torch.float64

a = torch.FloatTensor([1, 2])
print(a.dtype)  # torch.float32
b = torch.IntTensor([[1, 2], [3, 4]])
print(b.dtype)  # torch.int32
c = torch.LongTensor([2, 3])
print(c.dtype)  # torch.int64
d = torch.DoubleTensor([2, 3])
print(d.dtype)  # torch.float64

# 快速生成
print(torch.eye(3, 3))  # 3阶单位张量
print(torch.ones(4, 4, dtype=torch.long))
e = torch.ones(3, 4, dtype=torch.long)
print(e.dtype)  # torch.int64
print(torch.zeros(2, 2))
print(torch.empty(4, 3))  # 并不是0向量，也不是空矩阵
print(torch.randn(2, 3))
# 查看tensor形状，shape、size（）
print(e.shape)  # torch.Size([3, 4])
print(e.size())  # torch.Size([3, 4])
print(e.size(1))  # 4
print(e.size(0))  # 3
# 重塑tensor形状，view（）
print(e.view(2, 6))
print(e.view(4, -1))  # 原长度为12，故这里的-1默认为3
print(e.view(2, -1))  # 原长度为12，故这里的-1默认为6

# tensor中只有一个数值时，可以使用item（）提取出来变成数值
x = torch.Tensor([3])
print(x)  # tensor([3.])
print(type(x.item()))  # <class 'float'>
print(x.item())  # 3.0

# 从tensor转成numpy
x = torch.Tensor([[1, 2], [3, 4]])
print(x)
print(x.numpy())
print(type(x.numpy()))  # <class 'numpy.ndarray'>
# 从numpy转成tensor
y = x.numpy()
y = torch.from_numpy(y)
print(y)
print(type(y))  # <class 'torch.Tensor'>

# tensor的运算
x = torch.Tensor([[1, 2], [3, 4]])
y = torch.Tensor([[2, 2], [2, 2]])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x % y)
print(x.add_(y))
# x = x.add(y)
print(x)  # x = x.add_(y)
print(torch.add(x, y, out=x))  # x = x.add_(y) + y
print(x)  # x = x.add_(y) + y
# 乘法mul,对位相乘叫做点积
print(x.mul(y))  # x = x.add_(y) * y

x = torch.Tensor([[1,2,3]])
y = torch.Tensor([[1],
                  [2],
                  [3]])
#矩阵的乘法（a，n）(n,b)
print(x.mm(y))
print(torch.mm(x,y))
print(torch.mm(y,x))
