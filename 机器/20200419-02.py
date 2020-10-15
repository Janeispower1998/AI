'''
    使用回归算法预测波士顿房价
'''

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

boston = load_boston()
X = boston.data
y = boston.target
print(X[:5])
#将数据标准化
ss = StandardScaler()
X = ss.fit_transform(X)
#拆分数据集
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

#1.创建KNN模型
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(x_train,y_train)

#2.创建线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

#3.创建非线性支持向量机
from sklearn.svm import SVR
svr = SVR()
svr.fit(x_train,y_train)

#4.创建线性支持向量机
from sklearn.svm import LinearSVR
lsvr = LinearSVR()
lsvr.fit(x_train,y_train)


#5.创建决策树模型
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)

#6.随机森林模型
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)

#使用各个模型对测试集进行预测，计算误差
knn_res  = knn.predict(x_test)
lr_res =lr.predict(x_test)
svr_res = svr.predict(x_test)
lsvr_res = lsvr.predict(x_test)
dtr_res = dtr.predict(x_test)
rfr_res = rfr.predict(x_test)

# print(knn_res)
# print(svr_res)
# print(y_test)

#均方误差，绝对值误差，r2方法
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print('---------------k近邻-------------')
print('均方误差：',mean_squared_error(y_test,knn_res))
print('绝对值误差：',mean_absolute_error(y_test,knn_res))
print('r2:',r2_score(y_test,knn_res))
print('---------------线性回归-------------')
print('均方误差：',mean_squared_error(y_test,lr_res))
print('绝对值误差：',mean_absolute_error(y_test,lr_res))
print('r2:',r2_score(y_test,lr_res))
print('---------------非线性支持向量机-------------')
print('均方误差：',mean_squared_error(y_test,svr_res))
print('绝对值误差：',mean_absolute_error(y_test,svr_res))
print('r2:',r2_score(y_test,svr_res))
print('---------------线性支持向量机-------------')
print('均方误差：',mean_squared_error(y_test,lsvr_res))
print('绝对值误差：',mean_absolute_error(y_test,lsvr_res))
print('r2:',r2_score(y_test,lsvr_res))
print('---------------决策树-------------')
print('均方误差：',mean_squared_error(y_test,dtr_res))
print('绝对值误差：',mean_absolute_error(y_test,dtr_res))
print('r2:',r2_score(y_test,dtr_res))
print('---------------随机森林-------------')
print('均方误差：',mean_squared_error(y_test,rfr_res))
print('绝对值误差：',mean_absolute_error(y_test,rfr_res))
print('r2:',r2_score(y_test,rfr_res))






