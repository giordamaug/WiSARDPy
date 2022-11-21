# -*- coding: utf-8 -*-
"""
=========================================================
Linear Regression Example
=========================================================
The example below uses only the first feature of the `diabetes` dataset,
in order to illustrate the data points within the two-dimensional plot.
The straight line can be seen in the plot, showing how linear regression
attempts to draw a straight line that will best minimize the
residual sum of squares between the observed responses in the dataset,
and the responses predicted by the linear approximation.

The coefficients, residual sum of squares and the coefficient of
determination are also calculated.

"""

# Code source: Jaques Grobler
# License: BSD 3 clause

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as ltb
from regressor import WiSARDRegressor
from regression import WiSARDRegressor as WiSARDRegressor2
from utilities import binarize

# Load the diabetes dataset
df = pd.read_csv('datasets/bottle.csv')
df_binary = df[['Salnty', 'STheta', 'T_degC']]

# Taking only the selected two attributes from the dataset
df_binary.columns = ['Sal', 'STheta', 'Temp']
#display the first 5 rows
print(df_binary.head())
df_binary.fillna(method ='ffill', inplace = True)
df_binary500 = df_binary[:][:500]
  
X = np.array(df_binary500[['Sal', 'STheta']]) #.reshape(-1, 1)
y = np.array(df_binary500['Temp'])

print(X.shape, y.shape)
  
df_binary500.dropna(inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

method = 'WNN2'
print(f'Regression with method= "{method}"')

if method == 'LR':
	regr = LinearRegression()
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)
elif method == 'LGBM':
	regr = ltb.LGBMRegressor()
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)
elif method == 'WNN2':
	regr = WiSARDRegressor2(n_bits=8, n_tics=256, random_state=0, mapping='random', code='t', scale=True, debug=True)
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)
elif method == 'WNN':
	size = X_train.shape[1]*64
	regr = WiSARDRegressor(nobits=8, size=size, seed=0, dblvl=1)
	X = binarize(X_train, size, 't')
	Xt = binarize(X_test, size, 't')
	regr.fit(X, y_train)
	y_pred = regr.predict(Xt)

#print(regr.score(X_test, y_test))
#sns.lmplot(x ="Sal", y ="Temp", data = df_binary500, order = 2, ci = None)

# The mean squared error
# The coefficient of determination: 1 is perfect prediction
tres = regr.score(X_train,y_train)
print(f"Train accuracy {round(tres*100,2)} %")
print(f"Mean squared error: {round(mean_squared_error(y_test, y_pred),3)}")
print(f"Coefficient of determination: {round(r2_score(y_test, y_pred)*100,2)} %")
plt.scatter(X_test[:,0], y_test, color ='b')
plt.plot(X_test[:,0], y_pred, color ='k')
  
plt.show()
