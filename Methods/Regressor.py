#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.metrics import r2_score

class Regressor(ABC):
    name      = 'Regression'
    _features = None
    
    @abstractmethod
    def fit(self, x, y):
        """Train model"""
        pass
    
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

    def plot(self, x, y, transformed=False, x_train=None, y_train=None):
        """plot a graph with test data"""
        if transformed or x.ndim == 1 or x.shape[1] == 1:
            x_pred = np.linspace(min(x), max(x), 100)
            y_pred = [self.predict(x) for x in x_pred]
            plt.grid()
            if x_train is not None and y_train is not None:
                plt.scatter(x_train, y_train, color='r', label='train_data')
            plt.scatter(x, y, color='b', label='test_data')
            plt.plot(x_pred, y_pred, color='green')
            plt.legend()
        else:
            pca = PCA(n_components=1)            
            x_std = StandardScaler().fit_transform(x)
            x  = pca.fit_transform(x_std)
            if x_train is not None and y_train is not None:
                x_train_std = StandardScaler().fit_transform(x_train)
                x_train = pca.fit_transform(x_train_std)
                plt.scatter(x_train, y_train, color='r', label='train_data')
            
            self.fit(x_train, y_train)
            x_pred = np.linspace(min(x), max(x), 100)
            y_pred = [self.predict(x) for x in x_pred]
            plt.scatter(x, y, color='b', label='test_data')
            plt.plot(x_pred, y_pred, color='green')
            plt.legend()
            
            return self.score(x, y)
    
    def score(self, x, y):
        """Score of trained model using coefficient of determination"""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        y_mean = np.mean(y)
        y_pred = [self.predict(x) for x in x]
        
        SS_res = np.sum((y - y_pred)**2)
        SS_tot = np.sum((y - y_mean)**2)
        
        return math.floor((1 - SS_res/SS_tot)*100) / 100
        
