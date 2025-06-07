# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from GNSS_Magnitude_V1.model import build_model6

# Tracking learning rate
def get_lr_metric(optimizer):
    def lr(y_true, y_esti):
        curLR = optimizer._decayed_lr(tf.float32)
        return curLR
    return lr

# Callbacks for training
def get_callbacks(model_dir):
    checkpoint_filepath = os.path.join(model_dir, 'model_weights.h5')
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=20, 
        verbose=1,
        mode='min', 
        restore_best_weights=True)
    
    return [checkpoint, early_stopping]

def trainer(nst, nt, nc, model_dir, case_nm, x_train, y_train, x_val, y_val):
    # Set random seed for reproducibility
    tf.random.set_seed(2)
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Training parameters
    batch_size = 16
    epochs = 200
    initial_learning_rate = 1e-2
    decay = 1e-1 / epochs
    
    print(f'\nTraining Configuration:')
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Initial learning rate: {initial_learning_rate}')
    print(f'Learning rate decay: {decay}\n')
    
    # Build and compile model
    model = build_model6(nst, nt, nc)
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate, decay=decay)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', get_lr_metric(optimizer)]
    )
    
    # Train model
    print('Starting training...')
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=get_callbacks(model_dir),
        shuffle=True
    )
    
    # Evaluate model
    loss, mae, lr = model.evaluate(x_val, y_val, verbose=0)
    print(f'\nValidation Results:')
    print(f'Loss: {loss:.5f}')
    print(f'MAE: {mae:.5f}')
    print(f'Learning Rate: {lr:.5f}\n')
    
    # Save the entire model
    model_path = os.path.join(model_dir, 'model.h5')
    model.save(model_path)
    print(f'Model saved to {model_path}')
    
    return model