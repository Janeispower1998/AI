#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    使用K近邻算法处理回归任务
'''
import numpy as np
import matplotlib.pyplot as plt
X1  = np.arange(1,101).reshape(100,-1)
print(X1)
Y1 = X1+np.random.uniform(-10,10,(100,1))

#plt.show()

from sklearn.neighbors import KNeighborsRegressor

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(X1,Y1)
plt.ion()
plt.show()
for i in range(1,31):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X1,Y1)
    res = knn.predict(X1)
    if i!=1:
        ax.lines.remove(line[0])
    line = plt.plot(X1,res,color = 'red')
    plt.pause(2)
plt.ioff()
plt.show()
