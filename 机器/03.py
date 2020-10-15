import numpy as np
import pandas as pd
breask_cancer=pd.read_csv(r'D:\python学习\breast_cancer.csv',header=None)
print(breask_cancer)
data=breask_cancer.loc[:,1:10]
print(data)
target=breask_cancer.loc[:,10]
print(target)
#进行数据处理，删除‘?’所在行
print(data.shape)
data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')
print(data.shape)

# 将数据集拆分为训练集和测试集
import sklearn
x_train,x_text,y_train,y_test=train_test_split(data,target,test_size=0.25,random_state=33)
