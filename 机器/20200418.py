#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    使用决策树和随机森林处理泰坦尼克号上人员信息
'''
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import  accuracy_score
titanic = pd.read_csv(r'D:\python学习\titanic.csv')
print(titanic.head())
data = titanic[['pclass','age','sex','survived']].copy()
print(data.head())
age_mean = data['age'].mean()
data['age'].fillna(age_mean,inplace = True)
print((data['age']==np.nan).sum())
#数据可视乎部分
# data = data.replace({'pclass':{'1st':1,'2nd':2,'3rd':3},'sex':{'male':0,'female':1}})
# data_survived = data[data['survived']==1]
# data_died = data[data['survived']==0]
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(data_survived['pclass'],data_survived['age'],data_survived['sex'],color='g',marker='^')
# ax.scatter(data_died['pclass'],data_died['age'],data_died['sex'],color='r',marker='o')
# ax.set_xlabel('pclass')
# ax.set_ylabel('age')
# ax.set_zlabel('sex')
# plt.show()

#

from sklearn.feature_extraction import DictVectorizer  #将字典转换成向量
X = data[['pclass','age','sex']]
X = X.to_dict(orient = 'record')
#sparse是否生成稀疏矩阵
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X)
# print(vec.get_feature_names())
# print(X)

#拆分数据集
x_train,x_test,y_train,y_test = train_test_split(X,data['survived'],test_size=0.25,random_state=33)

#生成决策树模型
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
res = dtc.predict(x_test)
#print(classification_report(res,y_test))
print(accuracy_score(res,y_test))
#生成随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=19)
rfc.fit(x_train,y_train)
res = rfc.predict(x_test)
# print(classification_report(res,y_test))
print(accuracy_score(res,y_test))