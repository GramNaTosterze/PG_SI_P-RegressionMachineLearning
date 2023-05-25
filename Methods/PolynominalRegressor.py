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
        def vector(n, i):
            x = np.zeros(n, dtype=int)
            x[i] = 1
            return x
        
        
        self.__thetaDesc = ['1']
        self.__featurePows = [[0]]
        x = np.array(x)
        x_orig = x
        y = np.matrix(y).T
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self._features = x.shape[1]
        # dodawanie dodatkowych rzędów
        
        x = np.hstack ((np.ones((x.shape[0], 1), dtype=x.dtype), x))
        
        generators = [vector(self._features+1, i) for i in range(self._features + 1)]
        
        powers = list(map(sum, itt.combinations_with_replacement(generators, self.__degree)))
        self.__powers = powers
        
        x = np.hstack(np.array([((x**p).prod(1)).reshape(-1,1) for p in powers]))
        
        
        # wz: Theta = (X^T * X)^-1 * X^T * y
        X = np.matrix(x)
        X_T = X.T
        P1 = X_T * X
        P2 = X_T * y
        self.__theta = P1.I * P2
        
    def predict(self, sample):
        """Make prediction for provided sample"""
        sample = np.hstack(([1], sample))
        
        return sum([coeff * (sample**p).prod() for p, coeff in zip(self.__powers, self.__theta)])[0,0]
        
    def print(self):
        """prints some info"""
        features = ''.join('x' if self._features == 1 else [f"x{i}{''if i == self._features - 1 else ', '}" for i in range(self._features)])
        print(f"f({features}) = {self.__theta[0, 0]}", end='')
        for i in range(1, self.__theta.shape[0]):
            print(f" + {self.__theta[i, 0]}*{self.__thetaDesc[i]}", end='')
        print()