# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

def get_data(path, to_drop=None):
    
    raw_dataset = pd.read_csv(os.path.join('Data', path), na_values='?', comment='\t',
                              sep=',', skipinitialspace=True)
    
    dataset = raw_dataset.dropna()
    if to_drop is not None:
        dataset.drop(to_drop, axis=1, inplace=True)
    return dataset.select_dtypes([np.number])


def inspect_data(dataset):
    print('Dataset shape:')
    print(dataset.shape)

    print('Tail:')
    print(dataset.tail())

    print('Statistics:')
    print(dataset.describe().transpose())

    #sns.pairplot(dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
    #plt.show()


def split_data(dataset):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    return train_dataset, test_dataset

def save_to_latex(dataset, path):
    with open(path, 'w') as f:
        f.write(dataset.head().to_latex(index=False))