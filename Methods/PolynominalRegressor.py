#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import itertools as itt
from Regressor import Regressor

class PolynominalRegressor(Regressor):
    name = 'PolynominalRegression'
    def __init__(self, degree=2):
        self.__theta  = None
        self.__degree = degree
        
    
    def fit(self, x, y):
        """Train model"""
        def add_col(a1, a2):
            return np.hstack([a1, a2.reshape(-1, 1)])
        def product(pows, x):
            prod = 1
            for pow_i in range(len(pows)):
                prod = prod * ( x[:, pow_i] ** [pows[pow_i]] )
            return prod
        
        self.__thetaDesc = ['1']
        self.__featurePows = [[0]]
        x = np.array(x)
        y = np.matrix(y).T
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self._features = x.shape[1]
        # dodawanie dodatkowych rzędów
        
        X = np.ones((x.shape[0], 1))
        
        for pows in itt.product(range(self.__degree+1), repeat=self._features):
            if (sum(pows) == 0 or sum(pows) > self.__degree):
                continue
            prod = product(pows, x)
            X = add_col(X, prod)
            
            # X column names
            def prt_if_nZ(i, power):
                return f"x{i}^{power} " if power != 0 else ''
            feature_name = ''.join([f"{prt_if_nZ(i, pows[i])}" for i in range(len(pows))])
            self.__thetaDesc.append(feature_name)
            self.__featurePows.append(pows)
        
        
        # wz: Theta = (X^T * X)^-1 * X^T * y
        X = np.matrix(X)
        X_T = X.T
        P1 = X_T * X
        P2 = X_T * y
        self.__theta = P1.I * P2
        
    def predict(self, sample):
        """Make prediction for provided sample"""
        #return theta[2,0]*sample[0]**2 + theta[1,0]*sample[0] + theta[0,0]
        
        pred_val = 0
        for pow_i in range(len(self.__featurePows)):
            pred_samp = self.__theta[pow_i, 0]
            for i in range(len(self.__featurePows[pow_i])):
                pred_samp = pred_samp * sample[i]**self.__featurePows[pow_i][i]
            pred_val += pred_samp
        return pred_val  
        
    def print(self):
        """prints some info"""
        features = ''.join('x' if self._features == 1 else [f"x{i}{''if i == self._features - 1 else ', '}" for i in range(self._features)])
        print(f"f({features}) = {self.__theta[0, 0]}", end='')
        for i in range(1, self.__theta.shape[0]):
            print(f" + {self.__theta[i, 0]}*{self.__thetaDesc[i]}", end='')
        print()