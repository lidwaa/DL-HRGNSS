import numpy as np
import os

output_dir = './data/GNSS_M3S_181/'
os.makedirs(output_dir, exist_ok=True)

# Dimensions attendues : (batch, 3, 181, 3)
xdata = np.random.randn(200, 3, 181, 3).astype(np.float32)
ydata = np.random.uniform(5.0, 7.5, size=(200,)).astype(np.float32)

# Enregistrement
np.save(os.path.join(output_dir, 'xdata.npy'), xdata)
np.save(os.path.join(output_dir, 'ydata.npy'), ydata)
