# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:31:43 2023

@author: Krzysiu
"""
import time
import Data
from RegressionTree import RegressionTree
from copy import deepcopy
datasets = [
    ['Data/CarPrice_Assignment.csv', 'price']
    ]
y_label = datasets[0][1] # tmp
data = Data.get_data(datasets[0][0])

copy_data = deepcopy(data)
copy_data.drop(data.columns.difference([y_label, 'horsepower']), axis=1, inplace=True),
Data.save_to_latex(copy_data, 'Tables/CarPrice.tex')

train_data, test_data = Data.split_data(data)

y_train = train_data[y_label].to_numpy()
x_train = train_data.drop(y_label, axis=1).to_numpy()

y_test = test_data[y_label].to_numpy()
x_test = test_data.drop(y_label, axis=1).to_numpy()


# test implementation
regressor = RegressionTree(min_sample_split=3, max_depth=3)

time_0 = time.time()
regressor.fit(x_train, y_train)
print(f"time: {time.time() - time_0}\n")

regressor.print_tree()

#regressor.plot(x_test, y_test)


### testy sklear

import time
import Data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

data = Data.get_data(datasets[0][0])
y_label = datasets[0][1]
train_data, test_data = Data.split_data(data)

# data
y_train = train_data[y_label].to_numpy().reshape(-1,1)
x_train = train_data['horsepower'].to_numpy().reshape(-1,1)

y_test = test_data[y_label].to_numpy().reshape(-1,1)
x_test = test_data['horsepower'].to_numpy().reshape(-1,1)


## linear regression

# regression
regressor = LinearRegression()
time_0 = time.time()
regressor.fit(x_train, y_train)
time_1 = time.time() - time_0

# test
y_pred = regressor.predict(x_test)
plt.scatter(x_test, y_test, color='b')
plt.plot(x_test, y_pred, color='k')
plt.savefig('Plots/lin.png')
plt.show()

print("\nlinear regression:")
print(regressor.intercept_, regressor.coef_)
print(f"time: {time_1}")
print(f"score: {regressor.score(x_test, y_test)}")

## polynomial regression

# regression
time_0 = time.time()
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(poly_features, y_train)
time_1 = time.time() - time_0

# test
x = poly.fit_transform(x_test)
y_pred = regressor.predict(x)
plt.scatter(x_test, y_test, color='b')
plt.plot(x_test, y_pred, color='k')
plt.savefig('Plots/poly.png')
plt.show()

print("\npolynominal regression:")
print(regressor.intercept_, regressor.coef_)
print(f"time: {time_1}")
print(regressor.score(x, y_test))

## regression tree

# regression
regressor = DecisionTreeRegressor(random_state = 0) 
time_0 = time.time()
regressor.fit(x_train, y_train)
time_1 = time.time() - time_0

# test
y_pred = regressor.predict(x_test)
X_grid = np.arange(min(x_test), max(x_test), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(x_test, y_test, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.savefig('Plots/tree.png')
plt.show()

print("\ndecision tree regression:")
print(f"time: {time_1}")
print(f"score: {regressor.score(x_test, y_test)}")