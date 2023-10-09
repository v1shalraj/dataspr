import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
ii=pd.read_excel("C:\DS\EastWestAirlines.xlsx")
ii
target='Award'
IDcol = 'ID'
ii['Award'].value_counts()
np.random.randint
index=np.random.randint(low=0,high=len(ii),size=21755)
index.shape
index=pd.Series(index).unique()
index.shape
#divide into train and test sets
train_ii=ii.drop(index,axis=0)
test_ii=ii.loc[index]
print(train_ii.shape)
print(test_ii.shape)
#Choose all predictors except target & IDcols
predictors = [x for x in ii.columns if x not in ['Award']]
pd.Series(predictors)
gbm_v1 = GradientBoostingClassifier(random_state=10)
dtrain=train_ii.copy()
dtest=test_ii.copy()
 #Fit the algorithm on the data
gbm_v1.fit(dtrain[predictors], dtrain['Award'])
# Predict training set:
dtrain_predictions = gbm_v1.predict(dtrain[predictors])
dtrain_predprob = gbm_v1.predict_proba(dtrain[predictors])[:,1]
#Print model report:
print ("\nModel Report")
print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Award'].values, dtrain_predictions))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Award'], dtrain_predprob))
feat_imp = pd.Series(gbm_v1.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score');
#Perform cross-validation:
cv_score = cross_validate(gbm_v1, dtrain[predictors], dtrain['Award'], cv=5, scoring='roc_auc')
cv_score
print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score['test_score']),np.std(cv_score['test_score']),np.min(cv_score['test_score']),np.max(cv_score['test_score'])))
#Choose all predictors except target & IDcols
predictors = [x for x in ii.columns if x not in [target, IDcol]]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
                                  min_samples_leaf=50,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)
gsearch1.fit(ii[predictors],ii[target])
gsearch1.best_params_, gsearch1.best_score_
#Grid seach on subsample and max_features
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,
                                                max_features='sqrt', subsample=0.8, random_state=10), 
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4, cv=5)
gsearch2.fit(ii[predictors],ii[target])
gsearch2.best_params_, gsearch2.best_score_
#Grid seach on subsample and max_features
param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,
                                                    max_features='sqrt', subsample=0.8, random_state=10), 
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4, cv=5)
gsearch3.fit(ii[predictors],ii[target])
gsearch3.gsearch3.best_params_, gsearch3.best_score_
#Grid seach on subsample and max_features
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, 
                            min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10),
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4, cv=5)
gsearch4.fit(ii[predictors],ii[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#Grid seach on subsample and max_features
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, 
                            min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10, max_features=7),
                       param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(ii[predictors],ii[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
#Choose all predictors except target & IDcols
predictors = [x for x in ii.columns if x not in [target, IDcol]]
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200, 
                                         min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
#Choose all predictors except target & IDcols
predictors = [x for x in ii.columns if x not in [target, IDcol]]
gbm_tuned_2 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=9, min_samples_split=1200, 
                                         min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
