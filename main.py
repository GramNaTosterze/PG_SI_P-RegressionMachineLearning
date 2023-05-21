# -*- coding: utf-8 -*-
import os
import time
import Data
import sys
import matplotlib.pyplot as plt


sys.path.append('Methods')
from DecisionTreeRegressor import DecisionTreeRegressor
from PolynominalRegressor import PolynominalRegressor


datasets = [
    {'file_name': 'CarPrice_Assignment', 'y_label': 'price',  'to_drop': ['car_ID']},
    {'file_name': 'Cellphone', 'y_label': 'Sale', 'to_drop': ['Product_id']},
    #{'file_name': '', 'y_label': '', 'to_drop': ['']}
    ]
for dataset in datasets:
    data = Data.get_data(f"{dataset['file_name']}.csv", dataset['to_drop'])

    Data.save_to_latex(data, os.path.join('Tables',f"{dataset['file_name']}.tex"))

    train_data, test_data = Data.split_data(data)

    y_train = train_data[dataset['y_label']].to_numpy()
    x_train = train_data.drop(dataset['y_label'], axis=1).to_numpy()

    y_test = test_data[dataset['y_label']].to_numpy()
    x_test = test_data.drop(dataset['y_label'], axis=1).to_numpy()

    ## test implementation
    regressors = [PolynominalRegressor(degree=2), DecisionTreeRegressor(min_sample_split=3, max_depth=3)]
    for regressor in regressors:
        time_0 = time.time()
        regressor.fit(x_train, y_train)
        time_1 = time.time() - time_0
        print(f"name: {regressor.name}")
        print(f"time: {time_1}")
        print(f"score: {regressor.score(x_test, y_test)}")
        regressor.print()
        print()
        #regressor.fit_transform(x_train, y_train)
        regressor.plot(x_test, y_test)
        plt.savefig(os.path.join('Plots',f"{dataset['file_name']}_{regressor.name}.png"))
        plt.show()