#Importing required libraries
import pandas as pd 
import numpy as np

## Model Libraries
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# import libraries for metrics and reporting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import log_loss
#reading the data
data=pd.read_csv("C:\DS\Cars.csv")
#first five rows of the data
data.head()
#independent variables
x = data.drop(['MPG'], axis=1)

#dependent variable
y = data['MPG']
#import the train-test split
from sklearn.model_selection import train_test_split
#divide into train and test sets
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 101, stratify=y)
seed=1001
gbc = GradientBoostingClassifier(max_depth=1,n_estimators=101,warm_start=True,random_state=seed)
gbc.fit(train_x, train_y)
# Evaluation Train 
print("== Gradient Boosting ==")
print("Accuracy: {0:.2f}".format(accuracy_score(train_y, gbc.predict(train_x))))
print("Log loss: {0:.2f}".format(log_loss(train_y, gbc.predict_proba(train_x))))
# make predictions
gbc_y_pred = gbc.predict(test_x)
gbc_y_pred_prob = gbc.predict_proba(test_x)
# calculate log loss - Test
gbc_accuracy = accuracy_score(test_y, gbc_y_pred)
gbc_logloss = log_loss(test_y, gbc_y_pred_prob)

print("== Gradient Boosting ==")
print("Accuracy: {0:.2f}".format(gbc_accuracy))
print("Log loss: {0:.2f}".format(gbc_logloss))
#Importing XGBM Classifier 
from xgboost import XGBClassifier
#creating an extreme Gradient boosting instance
clf = XGBClassifier(random_state=96)
#training the model
clf.fit(train_x,train_y)
#calculating score on training data
clf.score(train_x, train_y)
#calculating score on test data
clf.score(test_x, test_y)
#set parameters
clf = XGBClassifier(random_state=96, colsample_bytree=0.7, max_depth=3)
#training the model
clf.fit(train_x,train_y)
#calculating score on training data
clf.score(train_x, train_y)
#calculating score on test data
clf.score(test_x, test_y)
clf = XGBClassifier(gamma=0.1, random_state=96)
#training the model
clf.fit(train_x,train_y)
#calculating score on test data
clf.score(test_x, test_y)
## conda install -c conda-forge lightgbm
import lightgbm as lgb
train_data=lgb.Dataset(train_x,label=train_y)
# parameters
params = {'learning_rate':0.01}
model= lgb.train(params,train_set=train_data)
y_pred=model.predict(test_x)
y_pred[:15]
