import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
ii=pd.read_excel("C:\DS\EastWestAirlines.xlsx")
ii
target='Award'
IDcol = 'ID'
ii['Award'].value_counts()
np.random.randint(1,10,1)
print(len(ii)*.25)
index=np.random.randint(0,len(ii),21755)
index.shape
index=pd.Series(index).unique()
index.shape
#divide into train and test sets
train_ii=ii.drop(index,axis=0)
test_ii=ii.loc[index]
print(train_ii.shape)
print(test_ii.shape)
test_results=pd.DataFrame(test_ii[['ID','Award']])
def modelfit(alg, dtrain, dtest, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    dtrain=train_ii.copy()
    dtest=test_ii.copy()
    predictors = [x for x in ii.columns if x not in [target, IDcol]]
    xgb_v1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    xgb_param = xgb_v1.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
    xgtest = xgb.DMatrix(dtest[predictors].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_v1.get_params()['n_estimators'], nfold=5,metrics='auc', early_stopping_rounds=101)
    xgb_v1.set_params(n_estimators=cvresult.shape[0])
#Fit the algorithm on the data
    xgb_v1.fit(dtrain[predictors], dtrain['Award'],eval_metric='auc')  
#Predict training set:
    dtrain_predictions = xgb_v1.predict(dtrain[predictors])
    dtrain_predprob = xgb_v1.predict_proba(dtrain[predictors])[:,1]
#Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Award'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Award'], dtrain_predprob))   
#     Predict on testing data:
    dtest['predprob'] = xgb_v1.predict_proba(dtest[predictors])[:,1]
    results = test_results.merge(dtest[['ID','predprob']], on='ID')
    print ('AUC Score (Test): %f' % metrics.roc_auc_score(results['Award'], results['predprob']))
    feat_imp = pd.Series(xgb_v1.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
    param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
}
    gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(ii[predictors],ii[target])
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
    param_test2 = {
    'max_depth':[4,5,6],
    'min_child_weight':[4,5,6]
}
    gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
                                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch2.fit(ii[predictors],ii[target])
    gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
    param_test2b = {
    'min_child_weight':[6,8,10,12]
}
    gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
                                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch2b.fit(ii[predictors],ii[target])
    gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_
#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
    param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}
    gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
                                        min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch3.fit(ii[predictors],ii[target])
    gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
    predictors = [x for x in ii.columns if x not in [target, IDcol]]
    xgb2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)modelfit(xgb2, train, test, predictors)
#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
                                        min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(ii[predictors],ii[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test5 = {
    'subsample':[i/100.0 for i in range(75,90,5)],
    'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
                                        min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(ii[predictors],ii[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
                                        min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(ii[predictors],ii[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test7 = {
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
                                        min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch7.fit(ii[predictors],ii[target])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
xgb3 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb3, train, test, predictors)
xgb4 = XGBClassifier(
        learning_rate =0.01,
        n_estimators=5000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb4, train, test, predictors)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(ii.drop('Award',axis=1))
scaled_features = scaler.transform(ii.drop('Award',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=ii.columns[:-1])
df_feat.head()
X_train, X_test, y_train, y_test = train_test_split(scaled_features,ii['Award'],
                                                    test_size=0.30)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
accuracy_rate = []

# May take some time
for i in range(1,40):
    ii
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,ii['Award'],cv=10)
    accuracy_rate.append(score.mean())
accuracy_rate
plt.figure(figsize=(10,6))

plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=3
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=3')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
# NOW WITH K=13, 14, 15, 16, 17, 18
knn = KNeighborsClassifier(n_neighbors=36)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=20')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))