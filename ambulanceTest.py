#python
#encoding=utf-8
#coding:unicode_escape
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix  #混淆矩阵
from sklearn.metrics import accuracy_score  #准确率判断
import graphviz

def LoadData(path,date):
    d1=pd.read_csv(path+date)
    d1.loc[:,"SUM_Field15"] = d1.loc[:,"SUM_Field15"].fillna(d1.loc[:,"SUM_Field15"].median())
    d1.loc[:,"c1"] = d1.loc[:,"c1"].fillna(0)
    d1.loc[:,"c2"] = d1.loc[:,"c2"].fillna(0)
    d1.loc[:,"c3"] = d1.loc[:,"c3"].fillna(0)
    d1.loc[:,"c4"] = d1.loc[:,"c4"].fillna(0)
    d1.loc[:,"c5"] = d1.loc[:,"c5"].fillna(0)
    d1.loc[:,"c0"] = d1.loc[:,"c0"].fillna(0)
    d1.loc[:,"agenull"] = d1.loc[:,"agenull"].fillna(0)
    d1.loc[:,"age10"] = d1.loc[:,"age10"].fillna(0)
    d1.loc[:,"age20"] = d1.loc[:,"age20"].fillna(0)
    d1.loc[:,"age30"] = d1.loc[:,"age30"].fillna(0)
    d1.loc[:,"age40"] = d1.loc[:,"age40"].fillna(0)
    d1.loc[:,"age50"] = d1.loc[:,"age50"].fillna(0)
    d1.loc[:,"age60"] = d1.loc[:,"age60"].fillna(0)
    d1.loc[:,"age70"] = d1.loc[:,"age70"].fillna(0)
    d1.loc[:,"age80"] = d1.loc[:,"age80"].fillna(0)
    d1.loc[:,"age90"] = d1.loc[:,"age90"].fillna(0)
    d1.loc[:,"age100"] = d1.loc[:,"age100"].fillna(0)
    d1.loc[:,"total_incidents"] = d1.loc[:,"total_incidents"].fillna(d1.loc[:,"total_incidents"].median())
    d1.loc[:,"PNT_COUNT"] = d1.loc[:,"PNT_COUNT"].fillna(d1.loc[:,"PNT_COUNT"].median())
    d1.loc[:,"PERCENTAGE"] = d1.loc[:,"PERCENTAGE"].fillna(d1.loc[:,"PERCENTAGE"].median())
    d1.loc[:,"FREQUENCY"] = d1.loc[:,"FREQUENCY"].fillna(d1.loc[:,"FREQUENCY"].median())
    x_data=d1.iloc[:,1:-1]
    y_data=d1.iloc[:,-1]
    return x_data,y_data
PATH="D:/now/prelimilary3/"
X_DATA,Y_DATA= LoadData(PATH,"categoryall.csv")
print(X_DATA)
print(Y_DATA)

X = X_DATA.iloc[:,1:2]
enc = OneHotEncoder(categories='auto').fit(X)
result = enc.transform(X).toarray()


newdata = pd.concat([X_DATA,pd.DataFrame(result)],axis=1)
newdata.drop(["LS0A11CD","LS0A11NM"],axis=1,inplace=True)
#2维
pcn= PCA(n_components=2)
pcn.fit(newdata)
x_data = pcn.transform(newdata)
print(x_data)

#划分
x_train,x_test,y_train,y_test = train_test_split(
    x_data,Y_DATA,test_size = 0.2
 )


rfc1 = RandomForestRegressor (
                             n_estimators=33,
                             random_state = 0,
                             min_samples_split= 10,
                             min_samples_leaf = 4 ,
                             max_features = 21,
                             min_impurity_decrease = 0.1,
                             oob_score = True
                             )

rfc1.fit(x_train,y_train)#训练模型
result_y = rfc1.score(x_test,y_test)#测试结果

pre_y = rfc1.predict(x_test)#测试结果

# print(pcn.explained_variance_) # 各个新轴上的方差,即XX^T各个特征值
# print(pcn.explained_variance_ratio_) # 方差解释百分比
# print(pcn.components_) # 各个特征值对应特征向量(标准化的),也叫主成分系数
# res_ba=pcn.components_
# superpa = []
# for i in range(0,27,1):
#     print(i)
#     superpa.append(res_ba)
# print(max(superpa),superpa.index(max(superpa)))
# plt.figure(figsize=[20,5])
# plt.plot(range(0,27,1),superpa)
# plt.show()
#
# print(pre_y)
# print(result_y)#准确率