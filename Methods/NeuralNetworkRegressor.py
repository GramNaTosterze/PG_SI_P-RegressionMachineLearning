#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from Regressor import Regressor
from torch.utils.data import Dataset, DataLoader    

class Dataclass(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        
    def forward(self, x):
        return self.layers(x)


class NeuralNetworkRegressor(Regressor):
    name = 'NeuralNetworkRegression'
    def __init__(self, iterations, learning_rate):
        self.__learning_rate = learning_rate
        self.__epochs = 5
    
    def fit(self, x, y):
        """Train model"""
        # prep
        dataset = Dataclass(x, y)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
        mlp = MLP(x.shape[1])
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.__learning_rate)
        
        # train
        for epoch in range(0, self.__epochs):
            print(f'Starting epoch {epoch+1}')
            
            current_loss = 0.0
            
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))
      
                # Zero the gradients
                optimizer.zero_grad()
      
                # Perform forward pass
                outputs = mlp(inputs)
      
                # Compute loss
                loss = loss_function(outputs, targets)
      
                # Perform backward pass
                loss.backward()
      
                # Perform optimization
                optimizer.step()
      
                # Print statistics
                current_loss += loss.item()
                if i % 10 == 0:
                    print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                    current_loss = 0.0
        print("end")
        
        
    def predict(self, sample):
        """Make prediction for provided sample"""
        return 
    
    def print(self):
        """prints some info"""
        print()

