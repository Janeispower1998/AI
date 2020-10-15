#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    处理鸢尾花数据集，多分类
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

#加载数据
data_iris = load_iris()
X = data_iris.data
y = data_iris.target
print(X.shape)
print(y.shape)
print(np.unique(y))

#将数据集拆分，训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#将数据标准化
# ss = StandardScaler()
# x_train = ss.fit_transform(x_train)
# x_test =ss.transform(x_test)

#创建K近邻模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn_pred = knn.predict(x_test)
print('knn_pred:',accuracy_score(y_test,knn_pred))

#创建逻辑回归模型
#multi_class = 'ovr','multinomial','auto'
lr = LogisticRegression(solver='sag',multi_class='ovr')
lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)
print('lr_pred:',accuracy_score(y_test,lr_pred))

#创建svc模型,C,gamma
#decision_function_shape选择多分类方式，‘ovo’
svc = SVC(max_iter=10000)
svc.fit(x_train,y_train)
svc_pred = svc.predict(x_test)
print('svc_pred:',accuracy_score(y_test,svc_pred))

#创建LinearSVC,
#LinearSVC,multi_class='crammer_singer','ovr'
lsvc = LinearSVC(multi_class='crammer_singer')
lsvc.fit(x_train,y_train)
lsvc_pred=lsvc.predict(x_test)
print('lsvc_pred:',accuracy_score(y_test,lsvc_pred))

