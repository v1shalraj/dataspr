import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
aa=pd.read_csv("C:\DS\Invistico_Airline.csv")
aa
def object_cols(df):
    return list(df.select_dtypes(include='object').columns)

def numerical_cols(df):
    return list(df.select_dtypes(exclude='object').columns)
obj_col = object_cols(aa)
num_col = numerical_cols(aa)
num_col
aa['Arrival Delay in Minutes'] = aa['Arrival Delay in Minutes'].fillna(aa['Arrival Delay in Minutes'].median())
aa.isna().sum()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
le = LabelEncoder()
norm = Normalizer()
for col in obj_col:
    aa[col] = le.fit_transform(aa[col])
aa.isna().sum()
from sklearn.model_selection import train_test_split
X_aa = aa.drop(['satisfaction'], axis = 1)
y_aa = aa['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X_aa, y_aa, test_size=0.33, random_state=42)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
fit_model = log_reg.fit(X_train, y_train)
preds = fit_model.predict(X_test)
preds
probs = fit_model.predict_proba(X_test)
probs
model_results = pd.DataFrame([preds, y_test, [elem[0] for elem in probs], [elem[1] for elem in probs]])
model_results = model_results.T
model_results.rename(columns = {0 : 'PredictedClass', 1 : 'TrueClass', 2 : 'ClassProb:0', 3 : 'ClassProb:1'}, inplace = True)
model_results

model = LogisticRegression(penalty='l2', C=1.0, random_state= 7, solver='lbfgs')
model.fit(X_train, y_train)
y_pred_class=model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)
y_pred_class[:5]
y_pred_prob[:5,:]
def draw_cm( actual, predicted ):
    cm = confusion_matrix( actual, predicted)
    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = [0,1] , yticklabels = [0,1] )
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.show()
draw_cm(y_test,y_pred_class);
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score,roc_curve
algo= []
tr = []
te = []
recall = []
precision = []
roc = []

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
# Visualize model performance with yellowbrick library
viz = ClassificationReport(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show();