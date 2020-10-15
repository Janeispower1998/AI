#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    KNN算法
    1.拆分数据集，进行交叉验证
    2.分类算法性能评估指标
    3.n_neighbors不同取值对模型的影响
'''
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  #手写数字图片，8*8，64个像素点
digits_data = load_digits()
# print(data)
x = digits_data.data #数据
print(x.shape)
#标签值
y = digits_data.target
print(y.shape)

# plt.ion()
# plt.show()
# for i in range(10): #取前十行数据
#     im = x[i].reshape(8,8) #将图片转换成8*8格式
#     plt.imshow(im,cmap='gray')
#     plt.pause(1)
# plt.ioff()
# plt.show()

#交叉验证，将数据集分为训练集和测试集两部分
from sklearn.model_selection import train_test_split
#test_size,若数值为（0，1）之间的数，表示测试集数据比例,若是大于1的数，测试集样本个数
#random_state 随机数标号，若为0或者不填，每次获取的随机数据集不同
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=100,random_state=33)
print(x_test.shape)

from sklearn.neighbors import KNeighborsClassifier
#metrics sklearn中性能评估模块
from sklearn.metrics import accuracy_score
# for i in range(1,31):
#     print('n_neighbors=',i)
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(x_train,y_train)
#     res = knn.predict(x_test)  #y_test
#     # 准确度，accuracy = 预测正确/样本总数
#     print('测试集准确度：',accuracy_score(y_test,res))
#     #knn模型预测训练集中的数据
#     res_train = knn.predict(x_train)
#     print('训练集准确度：',accuracy_score(y_train,res_train))

#accuracy = 预测正确/样本总数
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
res_train = knn.predict(x_train)  #预测值
#真实值y_train
import numpy as np
np.set_printoptions(threshold=np.inf)
# print(res_train)
# print('--------------------------------------')
# print(y_train)
print((res_train==y_train).sum()/len(y_train))
print(accuracy_score(y_train,res_train))

#分类算法，常用的评估指标     90%，0，100%，1，99%，2，70%
from sklearn.metrics import classification_report
print(classification_report(y_train,res_train))
#accuracy   准确度，
#precision  精确度，预测中真正的正类/预测成正类的数据之和
#recall    召回率,    预测出来的正类个数/数据集中所有正类
#f1-score
#suppor  每个类别中样本个数
#weighted avg 每个指标的加权平均数
#macro avg 宏平均值，每个指标的算数平均值
#micro avg 微平均值 = 准确度