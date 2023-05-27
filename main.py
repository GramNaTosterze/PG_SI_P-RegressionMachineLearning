# -*- coding: utf-8 -*-
import os
import time
import Data
import sys
import matplotlib.pyplot as plt
from tabulate import tabulate

# prep
sys.path.append('Methods')
if not os.path.exists('Tables'):
    os.mkdir('Tables')
if not os.path.exists('Plots'):
    os.mkdir('Plots')

# regressors
from DecisionTreeRegressor  import DecisionTreeRegressor
from PolynominalRegressor   import PolynominalRegressor
from LinearRegressor        import LinearRegressor
from NeuralNetworkRegressor import NeuralNetworkRegressor

datasets = [
    {'file_name': 'Student_Marks', 'y_label': 'Marks', 'title': 'Oceny otrzymane przez uczniów na bazie ilości kursów oraz godziń nauki', 'x_plt_label': 'czas nauki, ilość przedmiotów', 'y_plt_label': 'ocena', 'index': None},
    {'file_name': 'CarPrice_Assignment', 'y_label': 'price', 'title': 'Ceny samochodów na bazie różnych cech', 'x_plt_label': 'różne cechy samochodu', 'y_plt_label': 'cena samochodu', 'index': 'car_ID'}
    ]

for dataset in datasets:
    data = Data.get_data(f"{dataset['file_name']}.csv", dataset['index'])
    
    Data.save_to_latex(data, os.path.join('Tables',f"{dataset['file_name']}.tex"))
    
    train_data, test_data = Data.split_data(data)

    y_train = train_data[dataset['y_label']].to_numpy()
    x_train = train_data.loc[:, data.columns != dataset['y_label']].to_numpy()
    
    y_test = test_data[dataset['y_label']].to_numpy()
    x_test = test_data.loc[:, data.columns != dataset['y_label']].to_numpy()
    
    
    
    regressors = [LinearRegressor(), PolynominalRegressor(degree=3), DecisionTreeRegressor(min_sample_split=3, max_depth=4), NeuralNetworkRegressor()]
    for regressor in regressors:
        print()
        print(f"name: {regressor.name}")
        time_0 = time.time() 
        regressor.fit(x_train, y_train)
        time_1 = round(time.time() - time_0, 5)
        
        # test data
        #regressor.print()
        score_train = regressor.score(x_train, y_train)
        score_test  = regressor.score(x_test, y_test)
        
        plt.title( dataset['title'])
        plt.xlabel(dataset['x_plt_label'])
        plt.ylabel(dataset['y_plt_label'])
        score_PCA = regressor.plot(x_test, y_test, x_train=x_train, y_train=y_train)
        plt.savefig(os.path.join('Plots',f"{dataset['file_name']}_{regressor.name}.png"))
        plt.show()
        
        # save scores to table
        scores = [
                ['time', time_1],
                ['train score', score_train],
                ['test score', score_test],
                ['PCA score', score_PCA]
            ]
        print(tabulate(scores))
        with open(os.path.join('Tables', f"{dataset['file_name']}_{regressor.name}_scores.tex"), 'w') as f:
            f.write(tabulate(scores, tablefmt='latex', ).replace('lr', '|c|c|'))