#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import itertools as itt
from sklearn.decomposition import PCA

class PolynominalRegression:
    name = "Regresja wielomianowa"
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
        self.__features = x.shape[1]
        # dodawanie dodatkowych rzędów
        
        X = np.ones((x.shape[0], 1))
        
        for pows in itt.product(range(self.__degree+1), repeat=self.__features):
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
        self.__theta = (P1)**-1 * P2
        
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
        
    
    def plot(self, x, y):
        """plot a graph with test data"""
        if self.__features == 1:
            x_pred = np.arange(min(x), max(x), step=0.01)
            y_pred = [self.predict([x]) for x in x_pred]
            plt.scatter(x, y, color='b')
            plt.plot(x_pred, y_pred, color='green')
            #plt.show()
        else:
            pca = PCA(n_components=2)
            x_pca = pca.fit_transform(x, y)
            x = x_pca[:, 0]
            y = x_pca[:, 1]
            self.fit(np.array(x).reshape(-1,1),y)
            self.plot(x,y)
            
        
        
    def print(self):
        """prints some info"""
        features = ''.join('x' if self.__features == 1 else [f"x{i}{''if i == self.__features - 1 else ', '}" for i in range(self.__features)])
        print(f"f({features}) = {self.__theta[0, 0]}", end='')
        for i in range(1, self.__theta.shape[0]):
            print(f" + {self.__theta[i, 0]}*{self.__thetaDesc[i]}", end='')
        print()
    def score(self, x, y):
        """Score of trained model"""
        y_pred = [self.predict(x) for x in x]
        return ((y_pred - y)**2).mean()
        
        