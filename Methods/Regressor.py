#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
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

    def plot(self, x, y, transformed=False):
        """plot a graph with test data"""
        if transformed or x.ndim == 1 or x.shape[1] == 1:
            x_pred = np.arange(min(x), max(x), step=0.01)
            y_pred = [self.predict( [x for i in range(self._features)] ) for x in x_pred]
            plt.grid()
            plt.scatter(x, y, color='b')
            plt.plot(x_pred, y_pred, color='green')
        else:
            pca = PCA(n_components=2)
            test_data  = pca.fit_transform(x, y)
            x = test_data[:, 0]
            y = test_data[:, 1]
            #self.fit(np.array(x).reshape(-1,1), y)
            self.plot(x,y, transformed=True)
    
    def score(self, x, y):
        """Score of trained model using coefficient of determination"""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        y_mean = np.mean(y)
        y_pred = [self.predict(x) for x in x]
        
        SS_res = np.sum((y - y_pred)**2)
        SS_tot = np.sum((y - y_mean)**2)
        
        return round(1 - (SS_res/SS_tot), 2)
        
