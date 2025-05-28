# Deep Learning for GNSS-based Magnitude Estimation

## Project Overview and Technical Documentation

---

## 1. Project Overview

### Background

- Earthquakes require rapid and accurate magnitude estimation for emergency response
- Traditional seismic methods can saturate during large earthquakes
- GNSS (Global Navigation Satellite System) provides direct displacement measurements
- High-Rate GNSS data offers advantages over conventional seismic data

### Project Goals

- **Purpose**: Deep Learning model for real-time earthquake magnitude estimation using High-Rate GNSS data
- **Innovation**: Direct use of ground displacement data from GNSS stations
- **Advantages**: No saturation issues, suitable for large magnitude events

### Current Implementation

- **Input**: GNSS time series data from 3 stations (East, North, Up components)
- **Processing Window**: 181 seconds of data
- **Output**: Earthquake magnitude prediction (scale: 1-10)
- **Configuration**: GNSS_M3S_181 (3 stations, 181-second window)

---

## 2. Data Structure

### Input Data Format (xdata.npy)

- **Shape**: (N, 3, 181, 3) - 4-dimensional array
  - N: Number of earthquake samples in dataset
  - 3: Number of GNSS stations
  - 181: Time window length in seconds (3-minute window)
  - 3: Components per station (East, North, Up)

### Input Data Details

- **Sampling Rate**: 1 Hz (one measurement per second)
- **Units**: Displacement in meters
- **Time Window**: Captures full earthquake signal development
- **Station Selection**: 3 closest stations to epicenter

### Target Data (ydata.npy)

- **Shape**: (N,) - 1-dimensional array
- **Content**: Actual earthquake magnitudes
- **Range**: Typically between 5.0 and 9.5
- **Source**: Verified magnitude measurements

---

## 3. Project Architecture

### Core Components Overview

1. **main.py**:

   - Entry point and workflow orchestration
   - Handles data loading and model execution
   - Manages training and prediction pipeline

2. **GNSS_Magnitude_V1/**

   - **utils.py**:

     - Data loading and preprocessing
     - Train/validation/test split
     - Directory management

   - **training.py**:

     - Model training configuration
     - Learning rate scheduling
     - Early stopping implementation
     - Model checkpointing

   - **prediction.py**:

     - Model inference
     - Error calculation
     - Performance metrics

   - **model.py**:
     - Neural network architecture
     - Layer configuration
     - Activation functions

### Data Flow

1. Raw GNSS data → Preprocessing
2. Data split → Training/Validation/Test
3. Model training with validation
4. Model evaluation on test set
5. Magnitude prediction on new data

---

## 4. Neural Network Model

### Architecture Details

- **Input Layer**:
  - Shape: (3, 181, 3)
  - Accepts raw GNSS displacement data
- **Hidden Layers**:
  - Convolutional layers for feature extraction
  - Dense layers for magnitude estimation
  - Dropout layers for regularization
- **Output Layer**:
  - Single neuron
  - Linear activation
  - Predicts earthquake magnitude

### Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with custom learning rate
- **Batch Size**: Optimized for memory usage
- **Early Stopping**:
  - Monitors validation loss
  - Patience: 20 epochs
  - Prevents overfitting
- **Model Checkpointing**:
  - Saves best weights
  - Based on validation loss
  - Ensures optimal model selection

### Learning Rate Strategy

- Initial rate: 0.001
- Decay schedule: Reduce on plateau
- Minimum rate: 0.00001
- Helps convergence and prevents oscillation

---

## 5. Performance Metrics

### Current Model Performance

- **Mean Absolute Error**: ~0.076 magnitude units

  - Excellent accuracy for practical applications
  - Better than many traditional methods

- **Error Range**:
  - Minimum Error: 0.0 (perfect predictions)
  - Maximum Error: 0.6 magnitude units
  - RMSE: ~0.114 (robust performance metric)

### Validation Strategy

- **Data Split**:
  - 80% Training data
  - 10% Validation data
  - 10% Test set
- **Cross-Validation**:
  - Ensures model generalization
  - Prevents overfitting
  - Validates performance stability

### Real-world Implications

- Error < 0.5 magnitude units is excellent
- Suitable for emergency response
- Reliable for large magnitude events
- Fast computation for real-time use

---

## 6. Project Directory Structure

```
DL-HRGNSS/
├── data/                      # Data directory
│   └── GNSS_M3S_181/         # Current dataset
│       ├── xdata.npy         # Input features
│       └── ydata.npy         # Target magnitudes
├── GNSS_Magnitude_V1/        # Core package
│   ├── utils.py             # Utilities
│   ├── training.py          # Training logic
│   ├── prediction.py        # Prediction code
│   └── model.py             # Model architecture
├── output/                   # Results directory
│   └── model/               # Saved models
└── main.py                  # Entry point
```

### Key Files Explanation

- **xdata.npy**: Contains preprocessed GNSS time series
- **ydata.npy**: Contains corresponding magnitudes
- **model.py**: Defines neural network architecture
- **main.py**: Orchestrates the entire workflow

---

## 7. How to Use

### Environment Setup

```bash
# Clone repository
git clone [repository-url]

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install visualization tools
pip install pydot
winget install graphviz  # Windows
# apt-get install graphviz  # Linux
# brew install graphviz     # Mac
```

### Running the Model

```bash
# Basic usage
python main.py

# With specific configuration
python main.py --stations 3 --window 181
```

### Expected Output

- Model training progress
- Validation metrics
- Final test set performance
- Prediction results

---

## 8. Future Improvements

### Short-term Improvements

1. **Data Augmentation**

   - Synthetic data generation
   - Noise injection
   - Time-series augmentation

2. **Hyperparameter Optimization**
   - Automated tuning
   - Architecture search
   - Optimization algorithms

### Medium-term Goals

3. **Variable Station Support**

   - Dynamic input handling
   - Station weighting
   - Robust to missing data

4. **Real-time Capabilities**
   - Stream processing
   - Online learning
   - Quick inference

### Long-term Vision

5. **Web Interface**

   - User-friendly dashboard
   - Real-time monitoring
   - Result visualization
   - API access

6. **Advanced Features**
   - Uncertainty estimation
   - Early warning capabilities
   - Multi-parameter prediction

---

## 9. Questions & Discussion

### Key Points for Discussion

- Model performance requirements
- Deployment strategies
- Integration with existing systems
- Resource requirements
- Scaling considerations

Thank you for your attention!

Contact: [Your Contact Information]
