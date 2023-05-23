#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from Regressor import Regressor

class LinearRegressor(Regressor):
    name = 'LinearRegression'
    def __init__(self):
        self.__theta = None
        self._features = None

    def __standard(self, X, y):
        """Train model with standard method"""
        y = np.matrix(y).T
        X = np.column_stack((np.ones((len(X),1)), X))
        X = np.matrix(X)
        self.__theta = ((X.T * X).I) * X.T * y

    def fit(self, x, y):
        """Train model"""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self._features = x.shape[1]
        self.__standard(x, y)

    def predict(self, sample):
        """Make prediction for provided sample"""
        return self.__theta[0,0] + sum([self.__theta[i+1,0]*sample[i] for i in range(len(sample))])

    
    def print(self):
        """prints some info"""
        features = ''.join('x' if self._features == 1 else [f"x{i}{''if i == self._features - 1 else ', '}" for i in range(self._features)])
        print(f"f({features}) = {self.__theta[0,0]}", end='')
        for i in range(1, self.__theta.shape[0]):
            print(f" + {self.__theta[i,0]}*x{i-1}", end='')
        print()
