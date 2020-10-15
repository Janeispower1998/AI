from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

iris_data=load_iris()
x = iris_data.data
y = iris_data.target
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=33)


#数据标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) #选择多少个邻居
knn.fit(x_train,y_train)
res = knn.predict(x_test)
print("knn      :{}".format(accuracy_score(y_test,res)))

#逻辑回归
from sklearn.linear_model import LogisticRegression
#multi_class ovr(one vs rest),multinomial(many vs many),'auto'(自动选择一种多分类方式)
lr = LogisticRegression(multi_class='auto',solver='sag',max_iter=800)
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print("逻辑回归 :{}".format(accuracy_score(y_test,pred)))

#创建SVC模型
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(x_train,y_train)
result = clf.predict(x_test)
print("svc      :{}".format(accuracy_score(y_test,result)))

#创建LinearSVC模型
from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=10000)
clf.fit(x_train,y_train)
res = clf.predict(x_test)
print("LinearSVC:{}".format(accuracy_score(y_test,res)))
