# -*- coding: utf-8 -*-
import os
import time
import Data
import sys
import matplotlib.pyplot as plt

# prep
sys.path.append('Methods')
if not os.path.exists('Tables'):
    os.mkdir('Tables')
if not os.path.exists('Plots'):
    os.mkdir('Plots')

from DecisionTreeRegressor import DecisionTreeRegressor
from PolynominalRegressor import PolynominalRegressor
from LinearRegressor import LinearRegressor

datasets = [
    {'file_name': 'Student_Marks', 'y_label': 'Marks', 'x_labels': ['time_study']}, # score of a student according to hours spend studying
    {'file_name': 'possum', 'y_label': 'site', 'x_labels': ['belly']}, # age of possum based on their features
    {'file_name': 'CarPrice_Assignment', 'y_label': 'price',  'x_labels': ['horsepower', 'carlength', 'carwidth']}
    ] # do znalezienia najlepszej cechy: [list(itt.combinations(CECHY, r=i)) for i in range(1,MAX_CECH+1)] TODO
for dataset in datasets:
    data = Data.get_data(f"{dataset['file_name']}.csv")
    #Data.inspect_data(data)
    Data.save_to_latex(data, os.path.join('Tables',f"{dataset['file_name']}.tex"))

    train_data, test_data = Data.split_data(data)

    y_train = train_data[dataset['y_label']].to_numpy()
    x_train = train_data.loc[:, dataset['x_labels']].to_numpy()
    #x_train = train_data['horsepower'].to_numpy()
    
    y_test = test_data[dataset['y_label']].to_numpy()
    x_test = test_data.loc[:, dataset['x_labels']].to_numpy()
    #x_test = test_data['horsepower'].to_numpy()
    ## test implementation
    regressors = [LinearRegressor(), PolynominalRegressor(degree=3), DecisionTreeRegressor(min_sample_split=3, max_depth=4)]
    for regressor in regressors:
        time_0 = time.time()
        regressor.fit(x_train, y_train)
        time_1 = time.time() - time_0
        print(f"name: {regressor.name}")
        print(f"time: {time_1}")
        print(f"score_train: {regressor.score(x_train, y_train)}")
        print(f"score_test : {regressor.score(x_test, y_test)}")
        #regressor.print()
        print()
        
        if x_test.shape[1] != 1:
            regressor.fit_transform(x_train, y_train)
        regressor.plot(x_test, y_test)
        plt.savefig(os.path.join('Plots',f"{dataset['file_name']}_{regressor.name}.png"))
        plt.show()