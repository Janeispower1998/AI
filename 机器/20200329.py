#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
#聚类分析算法KMeans
from sklearn.cluster import KMeans
data = pd.read_csv(r'D:\python学习\航空公司客户价值分析\data\data_clean.csv')
print(data)
data = data.iloc[:,1:]
print(data.head())

#?将数据分为几类，也是需要根据不同的问题进行分析
k = 5
#创建KMeans模型
kmeans = KMeans(n_clusters=k)
#训练模型使用.fit()
kmeans.fit(data)
#查看质心坐标
print(kmeans.cluster_centers_)
#查看每条数据的分类情况
np.set_printoptions(threshold=np.inf)
print(kmeans.labels_)
print(len(kmeans.labels_))
r1 = pd.Series(kmeans.labels_,index = data.index)
r2 = r1.value_counts()
print(r2)

#为每行数据添加类别，数据+标签
r = pd.concat([data,r1],axis=1)
r.columns = list(data.columns)+['聚类类别']
#r.to_csv(r'D:\1-zr\晚班-02\20200328\kmeans_data_details.csv')
r3 = pd.DataFrame(kmeans.cluster_centers_)
print(r3)
r3.to_csv(r'D:\1-zr\晚班-02\20200328\kmeans_data_cluster.csv')







