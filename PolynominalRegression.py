#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:03:39 2023

@author: krzysiu
"""
import numpy as np

class PolynominalRegression:
    def __init__(self, degree=2):
        self.__omega  = None
        self.__degree = degree # TODO
    
    def fit(self, x, y):
        """Train model"""
        self.__standard(x, y)
        
    def __standard(self, x, y):
        """Train model with standard method"""
        x = np.array(x)
        y = np.matrix(y)
        X = np.concatenate((np.ones(x.shape), x, x*x)).reshape(3, -1)
        X = np.matrix(X)
        X_T = X.T
        x_dot_x = np.dot(X, X_T)
        X_inv = np.linalg.inv(x_dot_x)
        x_dot_y = np.dot(y, X_T)
        self.__omega = np.dot(X_inv, x_dot_y.T)
        
    def __GradientDescent(self, x, y):
        """Train model with Gradient Descent"""
        # TODO
        pass
        
    def predict(self, sample):
        """Make prediction for provided sample"""
        return self.__omega[0] + np.dot(self.__omega[1], sample) + np.dot(self.__omega[2], sample**2)
        pass
    def plot(self, test_data):
        """plot a graph with test data"""
        # TODO
        pass
    
    def print(self):
        """prints some info"""
        print(f"f(x) = {self.__omega[0]}*x^2 + {self.__omega[1]}x + {self.__omega[2]}")
    
    def score(self, sample):
        """Score of trained model"""
        # TODO
        pass