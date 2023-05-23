#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class Regressor(ABC):
    name      = 'Regression'
    _features = None
    
    @abstractmethod
    def fit(self, x, y):
        """Train model"""
        pass
        
    
    def fit_transform(self, x, y):
        """Train model and transfort to 2 featurer"""
        pca = PCA(n_components=2)
        train_data = pca.fit_transform(x, y)
        x = train_data[:, 0]
        y = train_data[:, 1]
        self.fit(np.array(x).reshape(-1,1),y)
    
    @abstractmethod
    def predict(self, sample):
        """Make prediction for provided sample"""
        pass
                
    @abstractmethod
    def print(self):
        """prints some info"""
        pass
    
    def model(self):
        return deepcopy(self)

    def best_featureset(self, features): # move from main
        """Returns a featureset with best score"""
        pass

    def plot(self, x, y, transformed=False, x_train=None, y_train=None):
        """plot a graph with test data"""
        if transformed or x.ndim == 1 or x.shape[1] == 1:
            x_pred = np.arange(min(x), max(x), step=0.01)
            y_pred = [self.predict( [x for i in range(self._features)] ) for x in x_pred]
            plt.grid()
            if x_train is not None and y_train is not None:
                plt.scatter(x_train, y_train, color='r', label='train_data')
            plt.scatter(x, y, color='b', label='test_data')
            plt.plot(x_pred, y_pred, color='green')
            plt.legend()
        else:
            pca = PCA(n_components=2)
            test_data  = pca.fit_transform(x, y)
            x = test_data[:, 0]
            y = test_data[:, 1]
            if x_train is not None and y_train is not None:
                train_data = pca.fit_transform(x_train, y_train)
                x_train = train_data[:, 0]
                y_train = train_data[:, 1]
            
            self.plot(x,y, transformed=True, x_train=x_train, y_train=y_train)
    
    def score(self, x, y):
        """Score of trained model using coefficient of determination"""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        y_mean = np.mean(y)
        y_pred = [self.predict(x) for x in x]
        
        SS_res = np.sum((y - y_pred)**2)
        SS_tot = np.sum((y - y_mean)**2)
        
        return round(1 - (SS_res/SS_tot if SS_res/SS_tot < 1 else 1), 2)
        
