# -*- coding: utf-8 -*-
import os
import time
import Data
import sys
import matplotlib.pyplot as plt
import numpy as np
import itertools as itt

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
    {'file_name': 'Student_Marks', 'y_label': 'Marks', 'x_labels': ['time_study']},
    {'file_name': 'CarPrice_Assignment', 'y_label': 'price',  'x_labels': ['horsepower', 'carlength', 'carwidth', 'peakrpm']}
    ]
def find_best_feature_combination(dataset, y_label, features, regressor):
    featureList = [list(itt.combinations(features, r=i)) for i in range(1,len(features)+1)]
    featureCombinations = list(itt.chain.from_iterable(featureList))
    
    y_train = dataset[y_label].to_numpy()
    
    best_model = None
    best_score = -np.inf
    best_featureSet = None
    
    for featureSet in featureCombinations:
        featureSet = list(featureSet)
        x_train = dataset.loc[:, featureSet].to_numpy()
        regressor.fit(x_train, y_train)
        score = regressor.score(x_train, y_train)
        if score > best_score:
            best_score = score
            best_model = regressor.model()
            best_featureSet = featureSet
    
    print(f"score_train: {best_score} - {best_featureSet}")
    regressor = best_model
    return best_featureSet


for dataset in datasets:
    data = Data.get_data(f"{dataset['file_name']}.csv")
    #Data.inspect_data(data)
    Data.save_to_latex(data, os.path.join('Tables',f"{dataset['file_name']}.tex"))

    train_data, test_data = Data.split_data(data)

    y_train = train_data[dataset['y_label']].to_numpy()
    #x_train = train_data.loc[:, dataset['x_labels']].to_numpy()
    
    y_test = test_data[dataset['y_label']].to_numpy()
    #x_test = test_data.loc[:, dataset['x_labels']].to_numpy()
    
    
    
    regressors = [LinearRegressor(), PolynominalRegressor(degree=3), DecisionTreeRegressor(min_sample_split=3, max_depth=4)]
    for regressor in regressors:
        print(f"name: {regressor.name}")
        featureSet = find_best_feature_combination(train_data, dataset['y_label'], dataset['x_labels'], regressor)
        x_train = train_data.loc[:, featureSet].to_numpy()
        time_0 = time.time() 
        regressor.fit(x_train, y_train)
        time_1 = time.time() - time_0
        
        # test data
        x_test = test_data.loc[:, featureSet].to_numpy()
        print(f"score_test : {regressor.score(x_test, y_test)}")
        print(f"time: {time_1}")
        
        regressor.print()
        print()
        
        if x_test.shape[1] != 1:
            regressor.fit_transform(x_train, y_train)
        regressor.plot(x_test, y_test, x_train=x_train, y_train=y_train)
        plt.savefig(os.path.join('Plots',f"{dataset['file_name']}_{regressor.name}.png"))
        plt.show()