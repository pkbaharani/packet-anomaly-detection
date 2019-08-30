#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 04:47:06 2019

@author: prateek
"""


import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


X1 = []
Xtestlist = []
bin_label= []
bin_label_test = []

dataset = pd.read_csv('~/Desktop/ACNS_Project3/NSL-KDD/KDDTrain+.txt',header=None)
X = dataset.iloc[:, 0:-2].values
label_column = dataset.iloc[:, -2].values



labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_1 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])

labelencoder_X_1 = LabelEncoder()
X[:,3] = labelencoder_X_1.fit_transform(X[:,3])
 
#print (X[:,:4]) 



#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
 
onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]







length=len(X)
for i in range(length):
    if(label_column[i]=='normal'):
        bin_label.append(0)
        X1.append(X[i,:])      
    else:
        if( label_column[i]=='neptune' or label_column[i]=='nmap'):
            bin_label.append(1)
            X1.append(X[i,:])
        else:
            if label_column[i]=='teardrop' or label_column[i]=='smurf'   :
                Xtestlist.append(X[i,:])
                bin_label_test.append(1)
                #print (X[i,:])
                



Xlearn= np.asarray(X1)
Xtest = np.asarray(Xtestlist)
ylearn= np.asarray(bin_label)
ytest= np.asarray(bin_label_test)

#print (X[:5,:])

sc = StandardScaler()
Xlearn = sc.fit_transform(Xlearn)
Xtest = sc.transform(Xtest)

#sc = MinMaxScaler(feature_range = (0, 1))

#Xlearn = sc.fit_transform(Xlearn)





classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(Xlearn[0])))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifierHistory = classifier.fit(Xlearn, ylearn, batch_size = 10, epochs = 25)



accavg = np.mean(classifierHistory.history['acc'])

y_pred = classifier.predict(Xtest)
y_pred = (y_pred > 0.9)
cm = confusion_matrix(ytest, y_pred)

testacc= (cm[0][0]+cm[1][1])/ (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

print ('Confusion Matrix:',cm)
 
plt.figure()
#plt.plot(classifierHistory.history.)   
plt.plot(classifierHistory.history['acc'] )   

plt.plot(classifierHistory.history['loss'] )   
plt.show()


print ('The average accuracy of the classifier:',accavg, 'Test Accuracy ',testacc)

