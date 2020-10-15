#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
    一、逻辑回归，处理多分类问题（二分类，是生成一条直线将两类数据分割开；多分类，需要生成多条直线，将各类数据分割开）
        多分类方式：
        1.one vs one ：每一个分类器在所有的类别中选择两个类别
        2.one vs rest：每个分类器，选择一类，剩下所有类归为一类  （逻辑回归）
        3.many vs many：将所有类别分为两类进行分割  （逻辑回归）
        多分类与二分类区别：
        1.数据集
        2.创建逻辑回归模型时，添加multi_class（可以控制多分类的执行方式）
    二、交叉验证（简单的交叉验证）
    三、数据标准化
'''
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
#手写数字8*8的图片，共10个类别
data = load_digits()
X = data.data
y = data.target
print(X.shape)
print(y.shape)
print(y)
y_se = pd.Series(y)
print(y_se.value_counts())

#数据标准化，z_score,归一化（0-1），每列的计算公式，（数据-均值）/方差
#消除单位限制，对列内各个数据分布形式不进行修改；使列间数据，具有相同规模
#标准化之后的数值，方差是1，均值是0
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
print(X[:5])
X = ss.fit_transform(X)  #先拟合后标准化
X = ss.transform(X)    #直接标准化
print(X[:5])

# #拆分数据集
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
# #训练模型
# #multi_class ovr(one vs rest),multinomial(many vs many),'auto'(自动选择一种多分类方式)
# lr = LogisticRegression(multi_class='ovr',solver='sag',max_iter=7000)
# lr.fit(x_train,y_train)
# pred = lr.predict(x_test)
# print(accuracy_score(y_test,pred))
#五折交叉验证
from sklearn.model_selection import cross_val_predict
lr = LogisticRegression(solver='sag',multi_class='ovr')
#交叉验证返回测试集验证结果的函数，cross_val_predict(模型，数据，标签，cv=n)n为n折交叉验证
# pred = cross_val_predict(lr,X,y,cv=5)
# np.set_printoptions(threshold=np.inf)
# print(pred)
# print(pred.shape)

#返回测试集评估指标,准确度，recall，精确度，F1
# from sklearn.model_selection import cross_val_score
# scores= cross_val_score(lr,X,y,cv=5,scoring='accuracy')
# print(scores)
# print(scores.mean())
#标准化是数据处理的方式之一，大多数情况下，使用数据标准化，会提高模型准确率
#可以返回多个评估指标的结果，返回训练集和测试集结果
from sklearn.model_selection import cross_validate
scoring = ['accuracy','recall_macro']
scores = cross_validate(lr,X,y,cv=5,scoring=scoring)
print(scores)
print(scores['test_accuracy'])