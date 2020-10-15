#!/usr/bin/env python
# -*- coding:utf-8 -*-

#乳腺癌肿瘤数据集
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC #可以是线性的也可以是非线性
from sklearn.svm import LinearSVC

#加载数据集
data = load_breast_cancer()
X = data.data
y = data.target
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=33)
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#创建SVC模型
#1.kernel核函数，默认rbf（高斯核函数），sigmoid，poly多项式核函数，linear线性核函数
#0.986013986013986,0.9790209790209791,0.972027972027972,0.916083916083916
#gamma,'rbf',更新公式中gamma的数值,默认值1/n_features,gamma 越小，支持向量个数越少，gamma越大，支持向量个数越多
#C，默认值是1，错误惩罚系数，C越大，对错误惩罚越大，C越小，对错误惩罚越小
#                  允许更少的点分类错误   允许更多的点分类错误
clf = SVC(kernel='rbf')
clf.fit(x_train,y_train)
result = clf.predict(x_test)
print(accuracy_score(y_test,result))

#可以查看支持向量的数据
print(clf.support_vectors_)
#查看支持向量的下标
print(clf.support_)
print(clf.n_support_)

#创建线性SVC分类器,没有kernel，support_vectors_，support_，n_support_
clf = LinearSVC(max_iter=10000)
clf.fit(x_train,y_train)
res = clf.predict(x_test)
print(accuracy_score(y_test,res))
print(clf.support_)