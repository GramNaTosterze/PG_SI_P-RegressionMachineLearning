# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:31:36 2023

@author: Krzysiu
"""
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, var_red=None, value=None):
        self.feature    = feature
        self.threshold  = threshold
        self.left       = left
        self.right      = right
        self.var_red    = var_red
        self.value      = value
        
        
class RegressionTree:
    def __init__(self, min_sample_split=2, max_depth=2):
        self.root               = None
        self.min_sample_split   = min_sample_split
        self.max_depth          = max_depth
    
    def fit(self, X, y):
        try:    # when features >= 2
            X.shape[1]
        except: # when only 1 feature
            X = np.array([X]).T # to not break other things when X is an 1d array
        self.root = self.split(X, y, 0)
    
    def split(self, X, y, depth):
        if depth == self.max_depth:
            return Node(value=np.mean(y)) # wynik
        
        node        = self.find_best_split(X, y)
        mask        = X[:, node.feature] < node.threshold
        node.left   = self.split(X[ mask], y[ mask], depth + 1)
        node.right  = self.split(X[~mask], y[~mask], depth + 1)
        return node
    
    def find_best_split(self, X, y) -> Node:
        best_feature, best_threshold = None, None
        best_score = np.inf # the lesser the better
    
        number_of_features = X.shape[1]
        for feature in range(number_of_features): 
            thresholds = np.unique(X[:,feature]).tolist()
            thresholds.sort()
            thresholds = thresholds[1:]
            
            for threshold in thresholds:
                mask    = X[:, feature] < threshold
                y_left  = y[ mask]
                y_right = y[~mask]
                t_rss   = self.rss(y_left, y_right)
                
                if t_rss < best_score:
                    best_score      = t_rss
                    best_threshold  = threshold
                    best_feature    = feature
        
        return Node(feature=best_feature, threshold=best_threshold)
    
    def predict(self, sample):
        prediction = None
        node = self.root
        while prediction is None:
            feature, threshold = node.feature, node.threshold
            if sample[feature] < threshold:
                node = node.left
            else:
                node = node.right
            prediction = node.value
        return prediction
    
    def rss(self, y1, y2):
        """Residual Sum of Squares - used to measure quality of split"""
        def srs(y):
            """Squared resudual sum"""
            return np.sum((y - np.mean(y))**2)
        
        return srs(y1) + srs(y2)
    
    def print_tree(self, tree=None, indent=" "):
        """Prints the tree"""
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    