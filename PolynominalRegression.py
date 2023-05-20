#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class PolynominalRegression:
    name = "Regresja wielomianowa"
    def __init__(self, degree=2):
        self.__theta  = None
        self.__degree = degree+1 # TODO
    
    def fit(self, x, y, Gradient=False):
        """Train model"""
        if Gradient:
            self.__GradientDescent(x, y)
        else:
            self.__standard(x, y)
        
    def __standard(self, x, y):
        """Train model with standard method"""
        x = np.array(x)
        y = np.matrix(y)
        
        # dodawanie dodatkowych rzędów
        
        X = np.concatenate((np.ones(x.shape), x))
        for i in range(2,self.__degree):
            X = np.concatenate((X, x**i))
        X = X.reshape(self.__degree, -1)
        # wz: Theta = (X^T * X)^-1 * X^T * y
        
        X = np.matrix(X)
        X_T = X.T
        self.__theta = (X * X_T)**-1 * (y * X_T).T
        
    def __GradientDescent(self, x, y):
        """Train model with Gradient Descent"""
        # TODO
        pass
        
    def predict(self, sample):
        """Make prediction for provided sample"""
        arr = [ self.__theta[i] * sample**i for i in range(self.__degree)]
        return np.array(arr).sum(axis=0).reshape(-1)
    
    def plot(self, x, y):
        """plot a graph with test data"""
        x_pred = np.arange(min(x), max(x), step=0.01)
        y_pred = self.predict(x_pred)
        plt.plot(x_pred, y_pred, color='b')
        
        plt.scatter(x, y, color='r')
        plt.show()
    
    def print(self):
        """prints some info"""
        print(f"f(x) = {self.__theta[0]}", end='')
        for i in range(1, self.__degree):
            print(f" + {self.__theta[i]}*x^{i}", end='')
        print()
    def score(self, sample):
        """Score of trained model"""
        # TODO
        pass