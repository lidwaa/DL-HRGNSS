# -*- coding: utf-8 -*-
import numpy as np
import os
import logging
import tensorflow as tf
import tensorflow.keras as keras
import statistics as statis
import shutil

# Prediction & Errors **************************************************************
def predict(x_test, y_test, model, dirout, case_nm):
    # Make predictions
    print('\nMaking predictions on test set...')
    pred = model.predict(x_test, verbose=0)
    y_pred = pred.reshape(len(y_test),)
    y_pred_rounded = np.array([round(val, 1) for val in y_pred])

    # Calculate errors
    errors = y_pred_rounded - y_test
    abs_errors = np.abs(errors)
    
    # Calculate statistics
    mean_error = np.mean(abs_errors)
    min_error = np.min(abs_errors)
    max_error = np.max(abs_errors)
    std_error = np.std(errors)
    mode_error = statis.mode(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    # Print results
    print('\nPrediction Results:')
    print('-' * 40)
    print(f'Mean Absolute Error : {mean_error:.3f}')
    print(f'Min Absolute Error  : {min_error:.3f}')
    print(f'Max Absolute Error  : {max_error:.3f}')
    print(f'Standard Deviation  : {std_error:.3f}')
    print(f'Mode Absolute Error : {mode_error:.3f}')
    print(f'RMSE               : {rmse:.3f}')
    print('-' * 40)
    
    # Print a few example predictions
    print('\nSample Predictions (first 5):')
    print('True Value -> Predicted Value (Error)')
    print('-' * 45)
    for i in range(min(5, len(y_test))):
        error = y_pred_rounded[i] - y_test[i]
        print(f'{y_test[i]:.1f} -> {y_pred_rounded[i]:.1f} ({error:+.1f})')
    print('-' * 45)