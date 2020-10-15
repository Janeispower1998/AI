#!/usr/bin/env python
# -*- coding:utf-8 -*-


'''
    使用逻辑回归算法，进行二分类
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
breask_cancer = pd.read_csv(r'D:\1-zr\晚班-01\20200317\breast_cancer.csv',header =None)
#print(breask_cancer.head())

#进行数据处理，删除’？’所在的行
print(breask_cancer.shape)
data = breask_cancer.replace(to_replace='?',value=np.nan)
data = data.dropna(how='any')
print(data.shape)

x = data.iloc[:,1:10]
target =data.loc[:,10]
#将数据集拆分成训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,target,test_size=0.25,random_state=33)

#创建线性回归模型
from sklearn.linear_model import LogisticRegression
#solver,选择更新参数使用何种方式，sag随机梯度下降，lbfas拟牛顿算法，newton-cg牛顿法
#max_iter  更新多少次参数
#tol误差小于tol时停止更新参数，默认值1e-4
lr = LogisticRegression(solver='sag',max_iter=3000)
lr.fit(x_train,y_train)
lrpredict = lr.predict(x_test)
# print(lrpredict)
# print(y_test)
print(accuracy_score(y_test,lrpredict))
from sklearn import metrics
print(metrics.confusion_matrix(y_test,lrpredict))
print(y_test.shape)
print((70+99)/(70+1+99+1))

res = lr.predict([[2	,1,	1,	1,	2,	1	,2,	1,	1]])
print(res)
res = lr.predict([[8	,7,	5	,10	,7,	9	,5	,5	,4]])
print(res)


