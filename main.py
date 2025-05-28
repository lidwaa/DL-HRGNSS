# -*- coding: utf-8 -*-
import logging
import warnings
import numpy as np
from GNSS_Magnitude_V1.utils import make_outfolders
from GNSS_Magnitude_V1.utils import make_data
from GNSS_Magnitude_V1.training import trainer
from GNSS_Magnitude_V1.prediction import predict
import os

# Fixed dataset configuration for GNSS_M3S_181
nt = 181  # Time window length in seconds
nst = 3   # Number of stations
nc = 3    # Number of components per station (East, North, Up)
case_nm = 'GNSS_M3S_181'  # Dataset name

# Minimal output directory structure
dir_out = './output'  # Main output directory
dir_model = f'{dir_out}/model'  # Directory for saving the trained model

def main():
    # Create minimal output directory
    os.makedirs(dir_model, exist_ok=True)
    
    # Load and split dataset
    print('Loading and splitting dataset...')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_data(None, case_nm)
    
    # Train model
    print('Training model...')
    model = trainer(nst, nt, nc, dir_model, case_nm, x_train, y_train, x_val, y_val)
    
    # Make predictions and print results
    print('Making predictions...')
    predict(x_test, y_test, model, None, case_nm)
    
    print('Processing completed successfully!')

if __name__ == '__main__':
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.ComplexWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        main()
