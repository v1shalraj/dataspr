
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
ii=sns.load_dataset("iris")
ii
ii.info()
ii.shape
ii.isna().sum()
ii['species'].value_counts()
ii.duplicated().sum()
duplicate_values = ii[ii.duplicated(keep=False)]

print("Duplicate Values:")
print(duplicate_values)
ii= ii.drop_duplicates()

print(ii)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
ii['species'] = label_encoder.fit_transform(ii['species'])
ii
y= ii['species']
X = ii.iloc[:,0:4]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=70)
X_train
from sklearn import tree, metrics, model_selection, preprocessing
from IPython.display import Image, display
dtree = tree.DecisionTreeClassifier(criterion="gini", max_depth=3,max_features='sqrt')
dtree.fit(X_train, y_train)
from sklearn import tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtree, 
                   feature_names=X_train.columns,  
                   class_names=['0','1','2'],
                   filled=True)
y_train_pred_class = dtree.predict(X_train)
pd.crosstab(y_train,y_train_pred_class)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train, y_train_pred_class)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
dtree.score(X_train,y_train)
dtree.score(X_test,y_test)
y_test_pred_class = dtree.predict(X_test)
pd.crosstab(y_test,y_test_pred_class)
from sklearn.ensemble import RandomForestClassifier
#creating a random forest instance
clf = RandomForestClassifier(criterion='gini',random_state=96,n_estimators = 150, max_depth=4)
#train the model
clf.fit(X_train,y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
y_train_pred_class = clf.predict(X_train)
y_test_pred_class = clf.predict(X_test)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train, y_train_pred_class)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
y_test_pred_class = clf.predict(X_test)
pd.crosstab(y_test,y_test_pred_class)

