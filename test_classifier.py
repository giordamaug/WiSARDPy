import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from lightgbm import LGBMClassifier
from regression import WiSARDClassifier
from sklearn.ensemble import RandomForestClassifier
import utilities as utils

# Load the dataset
df = pd.read_csv('datasets/iris.csv')
print(df.head())

labelname = 'species'
X = np.array(df.drop(labelname, axis=1))
le = preprocessing.LabelEncoder()
y = le.fit_transform(np.array(df[labelname]))
classes = le.classes_

print(f'Datasets dimensions: X={X.shape}, y={tuple(classes)}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

method = 'WNN'
print(f'Classification with method= "{method}"')

if method == 'LGBM':
	clf = LGBMClassifier()
elif method == 'WNN':
	clf = WiSARDClassifier(n_bits=16, n_tics=1024, random_state=0, mapping='random', code='t', scale=True, debug=True)
elif method == 'RF':
    clf = RandomForestClassifier()

y_pred = clf.fit(X_train, y_train).predict(X_test)
#y_pred = cross_val_predict(clf, X, y, cv=10)

print(f"Classification report: {metrics.classification_report(y, y_pred, target_names=classes)}")
plt.figure()
utils.plot_confusion_matrix(metrics.confusion_matrix(y, y_pred), classes=classes,title='Confusion matrix')
plt.show()
