import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from GNSS_Magnitude_V1.model import build_model6

# Create output directory for plots
os.makedirs('presentation_plots', exist_ok=True)

# Load data
data_dir = './data/GNSS_M3S_181/'
x_data = np.load(f'{data_dir}xdata.npy')
y_data = np.load(f'{data_dir}ydata.npy')

# 1. Data Structure Visualization
plt.figure(figsize=(12, 6))
sample_idx = 0
station_idx = 0
components = ['East', 'North', 'Up']
for i, comp in enumerate(components):
    plt.subplot(1, 3, i+1)
    plt.plot(x_data[sample_idx, station_idx, :, i])
    plt.title(f'{comp} Component')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
plt.suptitle('GNSS Components for Single Station', y=1.05)
plt.tight_layout()
plt.savefig('presentation_plots/data_structure.png')
plt.close()

# 2. Magnitude Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=y_data, bins=30)
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Count')
plt.savefig('presentation_plots/magnitude_distribution.png')
plt.close()

# 3. Sample Station Comparison
plt.figure(figsize=(15, 5))
for station in range(3):
    plt.subplot(1, 3, station+1)
    for comp in range(3):
        plt.plot(x_data[sample_idx, station, :, comp], 
                label=components[comp])
    plt.title(f'Station {station+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.legend()
plt.suptitle('Comparison of GNSS Components Across Stations', y=1.05)
plt.tight_layout()
plt.savefig('presentation_plots/station_comparison.png')
plt.close()

# 4. Model Architecture Visualization
plt.figure(figsize=(12, 8))
model = build_model6(nst=3, nt=181, nc=3)
tf.keras.utils.plot_model(model, to_file='presentation_plots/model_architecture.png', 
                         show_shapes=True, show_layer_names=True)
plt.close()

# 5. Training and Evaluation
# Split data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Compile and train model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='mse',
             metrics=['mae'])

history = model.fit(x_train, y_train, 
                   validation_data=(x_val, y_val),
                   epochs=10, batch_size=32, verbose=1)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.savefig('presentation_plots/training_history.png')
plt.close()

# 6. Prediction vs Actual
y_pred = model.predict(x_test, verbose=0)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Prediction Performance on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('presentation_plots/prediction_performance.png')
plt.close()

# 7. Error Distribution
errors = y_pred.flatten() - y_test
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30)
plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error (Predicted - Actual)')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('presentation_plots/error_distribution.png')
plt.close()

print("Enhanced plots have been generated in the 'presentation_plots' directory") 