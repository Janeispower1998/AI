import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
data=pd.read_csv(r'D:\python学习\航空公司客户价值分析\data\data_clean.csv')
data=data.iloc[:,1:]
print(data.head())
#把数据集分为几类 也是需要根据不同的问题进行分析
k=5
#创建KMeans模型
kmeans=KMeans(n_clusters=k)
#训练模型使用.fit()
kmeans.fit(data)
#查看质心坐标
print(kmeans.cluster_centers_)
#np.set_printoptions(threshold=np.inf)
print(len(kmeans.labels_))
print(kmeans.labels_)
r1=pd.Series(kmeans.labels_,index=data.index)
r2=r1.value_counts()
print(r2)
r1.name ='聚类名称'
r1=pd.concat([data,r1],axis=1)
#print(data)
r1.to_csv(r'D:\python学习\航空公司客户价值分析\data\kmeans_deta_details.csv')
r3=pd.DataFrame(kmeans.cluster_centers_)
print(r3)
r3.to_csv(r'D:\python学习\航空公司客户价值分析\data\kmeans_data_cluster.csv')
