# -*- coding: utf-8 -*-
import os
import time
import Data
from RegressionTree import RegressionTree
from PolynominalRegression import PolynominalRegression
from copy import deepcopy
datasets = [
    [os.path.join('Data', 'CarPrice_Assignment.csv'), 'price']
    ]
y_label = datasets[0][1] # tmp
data = Data.get_data(datasets[0][0])

copy_data = deepcopy(data)
copy_data.drop(data.columns.difference([y_label, 'horsepower']), axis=1, inplace=True),
Data.save_to_latex(copy_data, os.path.join('Tables','CarPrice.tex'))

train_data, test_data = Data.split_data(data)

y_train = train_data[y_label].to_numpy()
x_train = train_data['horsepower'].to_numpy()
#x_train = train_data.drop(y_label, axis=1).to_numpy()

y_test = test_data[y_label].to_numpy()
x_test = test_data['horsepower'].to_numpy()
#x_test = test_data.drop(y_label, axis=1).to_numpy()


## test implementation

regressors = [PolynominalRegression(3), RegressionTree(min_sample_split=3, max_depth=3)]

for regressor in regressors:
    time_0 = time.time()
    regressor.fit(x_train, y_train)
    time_1 = time.time() - time_0
    print(f"name: {regressor.name}")
    print(f"time: {time_1}")
    regressor.print()
    print(f"score: {'TODO'}")
    print()
    regressor.plot(x_test, y_test)
