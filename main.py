# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:31:43 2023

@author: Krzysiu
"""
import time
import Data
from RegressionTree import RegressionTree

data = Data.get_data()
train_data, test_data = Data.split_data(data)

y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()


# test implementation
regressor = RegressionTree(min_sample_split=3, max_depth=3)

time_0 = time.time()
regressor.fit(x_train, y_train)
print(f"time: {time.time() - time_0}")
regressor.print_tree()