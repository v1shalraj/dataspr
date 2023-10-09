import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
dd= pd.read_csv("C:\DS\diagnosis.csv")
dd
dd.shape
dd.info()   
dd.describe()
plt.boxplot(dd.radius_mean)
plt.boxplot(dd.texture_mean)
plt.boxplot(dd.perimeter_mean)
plt.boxplot(dd.area_mean)
plt.boxplot(dd.smoothness_mean)
plt.boxplot(dd.compactness_mean)
plt.boxplot(dd.concavity_mean)
plt.boxplot(dd.concave_points_mean)
plt.boxplot(dd.symmetry_mean)
plt.boxplot(dd.fractal_dimension_mean)
plt.boxplot(dd.radius_se)
plt.boxplot(dd.texture_se)
plt.boxplot(dd.perimeter_se)
plt.boxplot(dd.area_se)
plt.boxplot(dd.smoothness_se)
plt.boxplot(dd.compactness_se)
plt.boxplot(dd.concavity_se)
plt.boxplot(dd.concave_points_se)
plt.boxplot(dd.symmetry_se)
plt.boxplot(dd.fractal_dimension_se)
plt.boxplot(dd.radius_worst)
plt.boxplot(dd.texture_worst)
plt.boxplot(dd.perimeter_worst)
plt.boxplot(dd.area_worst)
plt.boxplot(dd.smoothness_worst)
plt.boxplot(dd.compactness_worst)
plt.boxplot(dd.concavity_worst)
plt.boxplot(dd.concave_points_worst)
plt.boxplot(dd.symmetry_worst)
plt.boxplot(dd.fractal_dimension_worst)
from feature_engine.outliers import Winsorizer
win= Winsorizer(capping_method='iqr',
                fold=0.05,
                tail='right',variables=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se',
                                        'area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
                                        'smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'])
dd2=win.fit_transform(dd[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se',
                        'area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
                        'smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst']])
col_of_replace=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se',
                        'area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
                        'smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst']
dd[col_of_replace]=dd2[col_of_replace]
dd
dd.drop(['id'],axis=1,inplace=True)

from sklearn.model_selection import train_test_split
X = dd.loc[:, dd.columns != 'diagnosis']  # independent variables

y = dd.loc[:, dd.columns == 'diagnosis']  # Target variable
X
X.head()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=70)
X_train.shape,X_test.shape
X_train.head()

from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score,roc_curve
algo= []
tr = []
te = []
recall = []
precision = []
roc = []
# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1.0, random_state= 7, solver='lbfgs')

model.fit(X_train, y_train)
y_pred_class=model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)
y_pred_class[:5]
y_pred_prob[:5,:]
## function to get confusion matrix in a proper format
def draw_cm( actual, predicted ):
    cm = confusion_matrix( actual, predicted)
    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = [0,1] , yticklabels = [0,1] )
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.show()
draw_cm(y_test,y_pred_class);
from sklearn.metrics import confusion_matrix


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_class)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for clarity
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_df=pd.DataFrame([fpr,tpr,thresholds]).T
roc_df.columns=['fpr','tpr','thresholds']
roc_df
algo.append('Logistic Regression')
tr.append(model.score(X_train, y_train))
te.append(model.score(X_test, y_test))
recall.append(recall_score(y_test,model.predict(X_test)))
precision.append(precision_score(y_test,model.predict(X_test)))
roc.append(roc_auc_score(y_test,model.predict(X_test)))
results = pd.DataFrame()
results['Model'] = algo
results['Training Score'] = tr
results['Testing Score'] = te
results['Recall'] = recall
results['Precision'] = precision
results['ROC AUC Score'] = roc
results = results.set_index('Model')
results
from yellowbrick.classifier import ClassificationReport, ROCAUC
# Visualize model performance with yellowbrick library
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show();
# Logistic Regression L1
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=7,C=10)

model_1.fit(X_train, y_train)
y_pred_class=model_1.predict(X_test)
y_pred_prob=model_1.predict_proba(X_test)
y_pred_class[:5]
y_pred_prob[:5,:]
## function to get confusion matrix in a proper format
def draw_cm( actual, predicted ):
    cm = confusion_matrix( actual, predicted)
    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = [0,1] , yticklabels = [0,1] )
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.show()
draw_cm(y_test,y_pred_class);

from sklearn.metrics import confusion_matrix


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_class)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for clarity
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_df=pd.DataFrame([fpr,tpr,thresholds]).T
roc_df.columns=['fpr','tpr','thresholds']
roc_df
algo.append('Logistic Regression')
tr.append(model.score(X_train, y_train))
te.append(model.score(X_test, y_test))
recall.append(recall_score(y_test,model.predict(X_test)))
precision.append(precision_score(y_test,model.predict(X_test)))
roc.append(roc_auc_score(y_test,model.predict(X_test)))
results = pd.DataFrame()
results['Model'] = algo
results['Training Score'] = tr
results['Testing Score'] = te
results['Recall'] = recall
results['Precision'] = precision
results['ROC AUC Score'] = roc
results = results.set_index('Model')
results
from yellowbrick.classifier import ClassificationReport, ROCAUC
# Visualize model performance with yellowbrick library
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show();
# Logistic Regression elastic net
from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression(penalty='elasticnet', solver='saga',C=10, random_state=7,)
y_pred_class=model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)
y_pred_class[:5]
y_pred_prob[:5,:]
## function to get confusion matrix in a proper format
def draw_cm( actual, predicted ):
    cm = confusion_matrix( actual, predicted)
    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = [0,1] , yticklabels = [0,1] )
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.show()
draw_cm(y_test,y_pred_class);

from sklearn.metrics import confusion_matrix


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_class)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for clarity
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_df=pd.DataFrame([fpr,tpr,thresholds]).T
roc_df.columns=['fpr','tpr','thresholds']
roc_df
algo.append('Logistic Regression')
tr.append(model.score(X_train, y_train))
te.append(model.score(X_test, y_test))
recall.append(recall_score(y_test,model.predict(X_test)))
precision.append(precision_score(y_test,model.predict(X_test)))
roc.append(roc_auc_score(y_test,model.predict(X_test)))
results = pd.DataFrame()
results['Model'] = algo
results['Training Score'] = tr
results['Testing Score'] = te
results['Recall'] = recall
results['Precision'] = precision
results['ROC AUC Score'] = roc
results = results.set_index('Model')
results
from yellowbrick.classifier import ClassificationReport, ROCAUC
# Visualize model performance with yellowbrick library
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show();