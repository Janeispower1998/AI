#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
#subset = 'all',train,test
news = fetch_20newsgroups(subset='all')
X = news.data#加载成一个列表，每个元素就是一个文章
y = news.target
print(len(X))
# print(news.filenames)
np.set_printoptions(threshold=np.inf)
print(np.unique(y))
print(X[0])
vec = TfidfVectorizer()
X = vec.fit_transform(X)
# print(X.shape)
# print(X[0])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

from sklearn.naive_bayes import MultinomialNB
for i in np.arange(0.01,1,0.1):
    mnb = MultinomialNB(alpha=i)
    mnb.fit(X_train,y_train)
    test_res  = mnb.predict(X_test)
# train_res = mnb.predict(X_train)

    print(accuracy_score(y_test,test_res))
# print(accuracy_score(y_train,train_res))

# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# lr.fit(X_train,y_train)
# pred = lr.predict(X_test)
# print(accuracy_score(y_test,pred))