# -*- coding: utf-8 -*-
import logging
import numpy as np
import os
from sklearn.model_selection import train_test_split

#
def make_outfolders(dir_out,dir_log,dir_info,dir_trmodel,dir_pred):
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    if not os.path.exists(dir_log):
        os.mkdir(dir_log)
        
    if not os.path.exists(dir_info):
        os.mkdir(dir_info)
        
    if not os.path.exists(dir_trmodel):
        os.mkdir(dir_trmodel)
    
    if not os.path.exists(dir_pred):
        os.mkdir(dir_pred)


# Load and split Data *********************************************************
def make_data(dir_info, dir_case):
    # Load data
    dirdata = f'./data/{dir_case}/'
    x = np.load(f'{dirdata}xdata.npy')
    y = np.load(f'{dirdata}ydata.npy')

    # Split into train+val and test sets (90% / 10%)
    ntest = 0.1
    ix = np.arange(0, len(y), 1, dtype=int)
    x_train1, x_test, ix_train1, ix_test = train_test_split(x, ix,
                                                           test_size=ntest,
                                                           random_state=1)
    y_train1 = y[ix_train1]
    y_test = y[ix_test]
    
    # Split train into train and validation sets (80% / 20% of training data)
    n_val = 0.2
    x_train, x_val, ix_train, ix_val = train_test_split(x_train1, ix_train1,
                                                       test_size=n_val,
                                                       random_state=1)
    y_train = y[ix_train]
    y_val = y[ix_val]
    
    # Save indices if dir_info is provided
    if dir_info is not None:
        dirout = f'{dir_info}/{dir_case}/'
        if not os.path.exists(dirout):
            os.makedirs(dirout)
        np.save(f'{dirout}index_datatrain.npy', ix_train)
        np.save(f'{dirout}index_dataval.npy', ix_val)
        np.save(f'{dirout}index_datatest.npy', ix_test)
    
    # Print dataset information
    print(f'\nDataset splits:')
    print(f'Training set   : {x_train.shape} samples')
    print(f'Validation set : {x_val.shape} samples')
    print(f'Test set      : {x_test.shape} samples\n')

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


