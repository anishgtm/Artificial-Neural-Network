# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:09:55 2019

@author: anish.gautam
"""

#TO PREDICT IF A PERSON WILL LEAVE A BANK OR NOT

import pandas as pd
import numpy as np
import keras

data=pd.read_csv('Churn_Modelling.csv')

X=data.iloc[:,3:13].values

y=data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])
X[:,2]=labelencoder.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Xtest=sc.transform(Xtest)

from keras.models import Sequential
from keras.layers import Dense

classifier= Sequential()

classifier.add(Dense(output_dim=6,input_dim=11,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=11,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(Xtrain,ytrain,batch_size=10,nb_epoch=5)

y_pred=classifier.predict(Xtest)
y_pred=ypred>0.5
ytest=ytest>0

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(ytest,y_pred)
acc=accuracy_score(ytest,y_pred)
print(acc)
