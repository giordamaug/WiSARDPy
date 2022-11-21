import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from lightgbm import LGBMClassifier
from wisard import WiSARDClassifier
from sklearn.ensemble import RandomForestClassifier
import utilities as utils
import time 
import argparse

parser = argparse.ArgumentParser(description='WiSARD in python')
parser.add_argument('-i', "--inputfile", dest='inputfile', metavar='<input file>', help='input file name', required=True)
parser.add_argument('-b', "--nbits", dest='nbits', metavar='<no of bits>', type=int, help='number of bits (default: 4)' , default=4, required=False)
parser.add_argument('-z', "--ntics", dest='ntics', metavar='<no of tics>', type=int, help='number of tics (default: 128)' , default=128, required=False)
parser.add_argument('-c', "--cvfold", dest='cvfold', metavar='<no of cv folds>', type=int, help='number of folds' , required=False)
parser.add_argument('-p', "--jobs", dest='jobs', metavar='<no of parallel jobs>', type=int, help='number of parallel jobs' , required=False)
parser.add_argument('-s', "--seed", dest='seed', metavar='<seed>', type=int, help='seed (default: 0)' , default=0, required=False)
parser.add_argument('-m', "--maptype", dest='maptype', metavar='<maptype>', type=str, help='mapping type (default: random, choice: random|linear)', choices=['random', 'linear'], default='random', required=False)
parser.add_argument('-M', "--method", dest='method', metavar='<method>', type=str, help='classifier name (default: RF, choice: RF|LGBM|WNN)', choices=['RF', 'WNN', 'LGBM'], default='WNN', required=False)
parser.add_argument('-D', "--debug", action='store_true', required=False)
parser.add_argument('-S', "--save-embedding", dest='saveembedding',  action='store_true', required=False)
parser.add_argument('-X', "--display", action='store_true', required=False)
args = parser.parse_args()


# Load the dataset
df = pd.read_csv(args.inputfile)
print(df.head())

labelname = 'species'
X = np.array(df.drop(labelname, axis=1))
le = preprocessing.LabelEncoder()
y = le.fit_transform(np.array(df[labelname]))
classes = le.classes_

print(f'Datasets dimensions: X={X.shape}, y={tuple(classes)}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

method = 'WNN'
print(f'Classification with method= "{method}"')

if method == 'LGBM':
	clf = LGBMClassifier()
elif method == 'WNN':
	clf = WiSARDClassifier(n_bits=args.nbits, n_tics=args.ntics, random_state=args.seed, mapping=args.maptype, code='t', scale=True, debug=args.debug)
elif method == 'RF':
    clf = RandomForestClassifier()

start = time.time()
if args.cvfold is None:
	y_pred = clf.fit(X_train, y_train).predict(X_test)
	targets = y_test
else:
	y_pred = cross_val_predict(clf, X, y, cv=args.cvfold)
	targets = y
print("--- %s seconds ---" % (time.time() - start))
print(f'{clf}')

print(f"Classification report: {metrics.classification_report(targets, y_pred, target_names=classes)}")
cm = metrics.confusion_matrix(targets, y_pred)
print(cm)
if args.display:
	plt.figure()
	utils.plot_confusion_matrix(cm, classes=classes,title='Confusion matrix')
	plt.show()
