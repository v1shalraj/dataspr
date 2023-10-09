import pandas as pd 
import numpy as np

import warnings
data=pd.read_csv("C:\DS\data_cleaned.csv")
data
data.head()
data.isnull().sum()
data.duplicated().sum()
data.describe().T
#independent variables
x = data.drop(['Survived'], axis=1)

#dependent variable
y = data['Survived']
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 101, stratify=y)
from sklearn.ensemble import AdaBoostClassifier
#creating an AdaBoost instance
clf = AdaBoostClassifier(random_state=96)
#training the model
clf.fit(train_x,train_y)
#calculating score on training data
clf.score(train_x, train_y)
#calculating score on test data
clf.score(test_x, test_y)
from sklearn.ensemble import RandomForestClassifier
clf = AdaBoostClassifier(random_state=96,n_estimators=500, learning_rate=0.21)
#training the model
clf.fit(train_x,train_y)
#calculating score on training data
clf.score(train_x, train_y)
#calculating score on test data
clf.score(test_x, test_y)
