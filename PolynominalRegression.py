#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:03:39 2023

@author: krzysiu
"""

class PolynominalRegression:
    __X
    __y
    def __init__(self):
        
    def __hypothesis(X, theta):
        y = theta*X
        return np.sum(y, axis=1)
    
    def __cost(X, y, theta):
        y1 = hypothesis(X, theta)
        m = len(y1)
        return sum(np.sqrt((y1-y)**2))/(2*m)
    
    def __gradientDescent(X, y, theta, alpha, epoch):
        J = []
        k = 0
        while k < epoch:
            y1 = hypothesis(X, theta)
            for c in range(0, len(X.columns)):
                theta[c] -= alpha*sum((y1-y)* X.iloc[:, c])/map(func, iter1)
    
    def fit(self):
        pass
    def predict(self):
        pass
    def plot(self):
        pass
    def score(self):
        """Score of trained model"""
        pass