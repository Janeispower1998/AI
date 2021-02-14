'''
一：
1,
输入层: 接收数据
隐藏层：传递信息
输出层：输出预测结果
2，
（反向传播 跟新参数 重复正向传播）
初始化权重 正向传播 计算误差 判断是否达到临界值
训练模型结束
3，
将低维空间非线性问题映射到高维空间编程线性问题进行处理 映射函数
4，
分类 回归
5，
监督学习 在学习过程中提供预测值 非监督学习 不提供
'''

#二：
#1,
import pandas as pd
from pandas import Series
Se = pd.Series([3.2,-3,2.2,4.1,-4])
Se[Se<0]=Se.mean();
print(Se)
print("==============")
#2
arr = [2,5,1,8,4]
arr1=[]
count=len(arr)-1
while count!=-1:
    arr1.append(arr[count])
    count-=1;
print(arr1)
print("==============")
#3
import numpy as np
arr_1=np.array([-1,0,1,2,3])
arr_2=np.ones((5,5))
print(arr_1+arr_2)
print("==============")

#4
arr_1=np.array(np.random.randn(10))
print(arr_1)
arr_1.sort(axis=0)
print(arr_1)

#三
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

iris_data=load_iris()
print(iris_data)
y = np.array(iris_data.target)
x = np.array(iris_data.data)
bo=(y==1) | (y==0)
y=y[bo]
x=x[bo]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1,random_state=33)

#生成随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=19)
rfc.fit(x_train,y_train)
res = rfc.predict(x_test)
print(accuracy_score(res,y_test))
