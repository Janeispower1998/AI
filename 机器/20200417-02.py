#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    监督学习，处理分类问题
    朴素贝叶斯算法，概率
    对文本类数据进行处理
'''

X = [
    'I would like to play computer games games',#文档,100
    'I would like to swimming',#100
    'I would like to play piano'
]

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(X)
#单词的统计
print(vec.get_feature_names())
#.toarray（）,正常矩阵
print(X.toarray())
#X是存储位置和数值，是一个稀疏矩阵
print(X)
'''
    tf-idf ,用于信息检索和数据挖掘中常用的加权技术
    tf 表示的是词频，词w在文档中出现的次数/文档中的总次数
    idf 逆文本频率指数 log_e(数据集中所有文档)/出现过词w的文档个数
    tf-idf = tf*idf
    
    降低在大多数文章中出现的单词的权重，增加重要单词的权重，重要的单词，在某一类文章中出现的频率更高
'''

X = [
    'I would like to play computer games games',#文档,100
    'I would like to swimming',#100
    'I would like to play piano'
]
from  sklearn.feature_extraction.text import  TfidfVectorizer
#min_df=n,如果n是整数，表示出现频次少于n次不进行统计，（0，1），在n%的文档中没有出现则不统计
vec_tfidf = TfidfVectorizer(min_df=2,max_df=0.8)
X = vec_tfidf.fit_transform(X)
print(vec_tfidf.get_feature_names())
print(X.toarray())
print(X)

