#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Neural Net modules
from keras.models   import Sequential
from keras.layers    import Dense, Dropout
from keras.callbacks import EarlyStopping

# regressor class
from Regressor import Regressor



class NeuralNetworkRegressor(Regressor):
    name = 'NeuralNetworkRegression'
    def __init__(self, epochs=5000, batch_size=50, print_additional_info=False):
        self.__epochs                = epochs
        self.__batch_size            = batch_size
        self.__verbose               = 1 if print_additional_info else 0
        self.__print_additional_info = print_additional_info
        self.__model                 = None
        self.__history               = None
    
    def fit(self, x, y):
        """Train model"""
        self.__model = Sequential()
        self.__model.add(Dense(1000, input_shape=(x.shape[1], ), activation='relu'))
        self.__model.add(Dense(500, activation='relu'))
        self.__model.add(Dense(250, activation='relu'))
        self.__model.add(Dense(1, activation='linear'))

        if self.__print_additional_info:
            self.__model.sumary()

        self.__model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        
        es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=50,
                   restore_best_weights = True)
        
        self.__history = self.__model.fit(x, y, 
                                   validation_data = (x,y),
                                   callbacks = [es],
                                   epochs = self.__epochs,
                                   batch_size = self.__batch_size,
                                   verbose = self.__verbose)
        
        
    def predict(self, sample):
        """Make prediction for provided sample"""
        model = self.__model
        return self.__model(sample.reshape(1,-1))[0,0]

    def print(self):
        """prints some info"""
        print()

