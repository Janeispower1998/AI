#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    将分类后的客户分类数据，使用雷达图进行可视化
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#步骤一：获取数据
data = pd.read_csv(r'D:\python学习\航空公司客户价值分析\data\kmeans_data_cluster.csv')
data = data.iloc[:,1:]
#保存两位小数
data = data.round(2)
print(data.head())
data['data0'] = data.iloc[:,0]
print(data.head())
fig = plt.figure()
ax = fig.add_subplot(111,polar=True)
#生成等差数列的函数linspace（起始，终点，个数）,endpoint,不包含右端点
# arr = np.linspace(0,2,5,endpoint=False)
# print(arr)
angels = np.linspace(0,2*np.pi,5,endpoint=False)
print(angels)
angels = np.append(angels,np.array(0))
print('*'*30)
print(angels)

for i,c in zip(range(5),['b','r','g','y','c']):
    plt.plot(angels,data.iloc[i],'o--',c=c,label = '第'+str(i)+'类数据')
labels = ['L','R','F','M','C']
ax.set_thetagrids(angels*180/np.pi,labels)
# print(angels*180/np.pi)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.legend()
plt.show()
print(np.pi)
