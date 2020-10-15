#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    监督学习，分类任务：
    KNN，K近邻
'''
import numpy as np
import matplotlib.pyplot as plt
#创建简单的数据
n_data = np.ones((100,2)) # 100行，2列，数据100条，每条有2列（2个属性）
x0 = np.random.normal(loc=2,scale=1,size=(100,2))#loc=2*n_data,scale=1)
print(x0)
y0 =np.zeros((100))  #标签
x1 = np.random.normal(loc=-2,scale=1,size=(100,2))#loc=-2*n_data,scale=1)
y1 = np.ones((100))

#将x0，x1两类数据融合成一个数据集,整个数据集有200条数据
x = np.append(x0,x1,axis=0)
print(x)
y = np.append(y0,y1,axis=0)
print(x.shape)
print(y.shape)
for i in range(len(x)):
    if y[i]==0:
        plt.scatter(x[i,0],x[i,1],c='red')
    else:
        plt.scatter(x[i,0],x[i,1],c='blue')

'''
    1.创建模型
    2.训练模型，fit()
    3.预测
    4.评估模型的性能（看好坏）
'''
#训练KNN模型，（无监督学习，kmeans）
from sklearn.neighbors import KNeighborsClassifier
#创建模型
knn = KNeighborsClassifier(n_neighbors=3) #选择多少个邻居
#训练
knn.fit(x,y)
#预测
pre_x0,pre_x1 = 0,-2
res = knn.predict([[pre_x0,pre_x1]])  #x, 有两列
print(res)

if res ==0:
    plt.scatter([pre_x0],[pre_x1],c='red',marker='^')
else:
    plt.scatter([pre_x0],[pre_x1],c='blue',marker='^')
plt.show()




