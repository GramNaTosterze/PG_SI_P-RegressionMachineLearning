#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers
from dataclasses import dataclass
from typing import List
from Regressor import Regressor

@dataclass
class FFN_Hyperparams:
    # stale (zalezne od problemu)
    num_inputs: int
    num_outputs: int

    # do znalezienia optymalnych (np. metodą Random Search) [w komentarzu zakresy wartości, w których można szukać]
    hidden_dims: List[int]              # [10] -- [100, 100, 100]
    activation_fcn: str                 # wybor z listy, np: ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'softplus']
    learning_rate: float                # od 0.1 do 0.00001 (losowanie eksponensu 10**x, gdzie x in [-5, -1])

class NeuralNetworkRegressor(Regressor):
    name = 'NeuralNetworkRegression'
    def __init__(self):
        self.__model = None
    
    def __train(self, train_data, experiment_dir, exp_dir='exp'):
        x, y = train_data
        es_cbk = tf.keras.callbacks.EarlyStopping(min_delta=0.1, patience=5)
        ckp_cbk = tf.keras.callbacks.ModelCheckpoint(os.path.join(experiment_dir, 'model_best_weights'), save_best_only=True, save_weights_only=True)
        tb_cbk = tf.keras.callbacks.TensorBoard()
        
        history = self.__model.fit(x=x, y=y, batch_size=64, validation_split=0.2,
                            epochs=100, verbose=1, callbacks=[es_cbk, ckp_cbk, tb_cbk])
        return history
    
    def __build_model(self, hp: FFN_Hyperparams):
        """Builds model"""
        model = tf.keras.Sequential()

        i = 0
        for dim in hp.hidden_dims:
            model.add(layers.Dense(dim, activation=hp.activation_fcn, input_shape=[hp.num_inputs], name=f'ukryta_{i}'))
            i += 1
        model.add(layers.Dense(hp.num_outputs, name='wyjsciowa'))  # model regresyjny, activation=None w warstwie wyjściowej
        
    
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.mse,
                      metrics=[tf.keras.metrics.mean_absolute_error, 'mse'])
        return model
    
    def __single_run(self, hp, train_data, experiment_dir):
        
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.__model = self.__build_model(hp)
        return self.__train(train_data, experiment_dir)
    
    def fit(self, x, y):
        """Train model"""
        
        experiment_dir = os.path.join('TrainData','exp')
        hp = FFN_Hyperparams(x.shape[1], 1, [10, 20, 10], 'relu', 0.001)

        # run training with them
        history = self.__single_run(hp, (x, y), experiment_dir)
        
        
    def predict(self, sample):
        """Make prediction for provided sample"""
        return self.__model.predict(sample)

    
    def print(self):
        """prints some info"""
        features = ''.join('x' if self._features == 1 else [f"x{i}{''if i == self._features - 1 else ', '}" for i in range(self._features)])
        print(f"f({features}) = {self.__theta[0,0]}", end='')
        for i in range(1, self.__theta.shape[0]):
            print(f" + {self.__theta[i,0]}*x{i-1}", end='')
        print()
