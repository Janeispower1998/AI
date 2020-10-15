#!/usr/bin/env python
# -*- coding:utf-8 -*-

#深度学习框架tensorflow（谷歌）、pytorch（facebook）、keras、caffe
import torch
#tensor张量，numpy，ndarray
#tensor创建
#a = torch.tensor([[1.0,2],[3,4]],dtype='')
# print(a.dtype)
a = torch.Tensor([[1,2],[2,3],[3,4]])  #默认是浮点类型
print(a)
print(type(a))
print(a.dtype)
a = a.int()
print(a.dtype)
a = a.long()
print(a.dtype)
a = a.double()
print(a.dtype)

a = torch.FloatTensor([1,2])
print(a.dtype)
b = torch.IntTensor([[1,2],[3,4]])
print(b.dtype)
c = torch.LongTensor([2,3])
print(c.dtype)
d = torch.DoubleTensor([2,3])
print(d.dtype)


#快速生成
print(torch.eye(3,3))
print(torch.ones(4,4,dtype=torch.long))
e = torch.ones(3,4,dtype=torch.long)
print(e.dtype)
print(torch.zeros(2,2))
print(torch.empty(4,3))
print(torch.randn(2,3))

#查看tensor形状，shape、size（）
print(e.shape)
print(e.size())
print(e.size(1))
#重塑tensor形状，view（）
print(e.view(2,6))
print(e.view(4,-1))
x = torch.Tensor([3])
print(x)
#tensor中只有一个数值时，可以使用item（）提取出来变成数值
print(type(x.item()))
x = torch.Tensor([[1,2],[3,4]])
print(x.numpy())
print(type(x.numpy()))
#从tensor转成numpy
y = x.numpy()
#从numpy转成tensor
y = torch.from_numpy(y)
print(y)
print(type(y))

#tensor的运算
x = torch.Tensor([[1,2],[3,4]])
y = torch.Tensor([[2,2],[2,2]])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x%y)
print(x.add_(y))
# x = x.add(y)
print(x)
print(torch.add(x,y,out = x))
print(x)
#乘法mul,对位相乘叫做点积
print(x.mul(y))
x = torch.Tensor([[1,2,3]])(2,2)
y = torch.Tensor([[1],
                  [2],
                  [3]
                    ])(3,1)
#矩阵的乘法（a，n）(n,b)
print(x.mm(y))
print(torch.mm(x,y))
print(torch.mm(y,x))#(3,1)(1,3)









