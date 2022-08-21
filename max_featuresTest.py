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
from sklearn.model_selection import cross_val_score
def LoadData(path,date):
    d1=pd.read_csv(path+date)
    d1.loc[:,"SUM_Field15"] = d1.loc[:,"SUM_Field15"].fillna(d1.loc[:,"SUM_Field15"].median())
    d1.loc[:,"c1"] = d1.loc[:,"c1"].fillna(d1.loc[:,"c1"].median())
    d1.loc[:,"c2"] = d1.loc[:,"c2"].fillna(d1.loc[:,"c2"].median())
    d1.loc[:,"c3"] = d1.loc[:,"c3"].fillna(d1.loc[:,"c3"].median())
    d1.loc[:,"c4"] = d1.loc[:,"c4"].fillna(d1.loc[:,"c4"].median())
    d1.loc[:,"c5"] = d1.loc[:,"c5"].fillna(d1.loc[:,"c5"].median())
    d1.loc[:,"c0"] = d1.loc[:,"c0"].fillna(d1.loc[:,"c0"].median())
    d1.loc[:,"total_incidents"] = d1.loc[:,"total_incidents"].fillna(d1.loc[:,"total_incidents"].median())
    d1.loc[:,"PNT_COUNT"] = d1.loc[:,"PNT_COUNT"].fillna(d1.loc[:,"PNT_COUNT"].median())
    d1.loc[:,"PERCENTAGE"] = d1.loc[:,"PERCENTAGE"].fillna(d1.loc[:,"PERCENTAGE"].median())
    d1.loc[:,"FREQUENCY"] = d1.loc[:,"FREQUENCY"].fillna(d1.loc[:,"FREQUENCY"].median())
    x_data=d1.iloc[:,1:-1]
    y_data=d1.iloc[:,-1]
    return x_data,y_data
PATH="D:/now/prelimilary2/"
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



superpa = []
rfc = RandomForestRegressor(n_estimators=25,n_jobs=-1,max_features = None,)
rfc_s = cross_val_score(rfc,x_data,Y_DATA,cv=10).mean()
print(rfc_s)
