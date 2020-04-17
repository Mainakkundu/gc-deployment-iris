# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:36:51 2020

@author: Mainak Kundu
"""

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd 


iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.20)
model =GradientBoostingClassifier(n_estimators=10)
model.fit(X_train,y_train)

pr = model.predict(X_test)
import pickle

with open(r'C:\Users\Mainak Kundu\Desktop\DEPLOYMENT-GCP\iris_gb.pkl','wb') as mdl:
    pickle.dump(model,mdl)
    

