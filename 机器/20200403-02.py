#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    非监督学习：K-means（将无标签的数据进行分类）
    非监督学习：数据降维（数据集中的列（特征、属性）），减少数据集中列的个数
    #注：不是单纯删除某列，而是通过较少的列表示较多的列
    数据降维的作用：
    1.减少数据量，加快运行效率
    2.减少相似特征，去除多余信息
    3.有的时候在消除噪音的时候，会使用降维
'''
#PCA算法，主成分分析
#分解：decomposition
from sklearn.decomposition import PCA
import pandas as pd
data = pd.read_csv(r'D:\python学习\digits.csv',header=None)
print(data.head())
data = data.iloc[:,0:64]
print(data.head())

print('---------------------pca----------------------------')
#创建PCA模型，
#n_components,将数据降到多少维
pca = PCA(n_components=8)
data = pca.fit(data).transform(data)
print(data.shape)
#方差的大小,数值越大，越重要，表达的信息越多
print(pca.explained_variance_)
print('--------------------------------------------------')
#可以查看方差占比
print(pca.explained_variance_ratio_)
#论文中，一般会选择占比之和在85%以上
print((pca.explained_variance_ratio_).sum())


