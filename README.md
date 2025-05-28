# Deep Learning for GNSS-based Magnitude Estimation

A deep learning model for real-time earthquake magnitude estimation using High-Rate GNSS data.

## Overview

This project implements a deep learning approach to estimate earthquake magnitudes using displacement data from GNSS stations. The model processes time series data from multiple GNSS stations to provide rapid and accurate magnitude estimates, which is crucial for early warning systems and emergency response.

## Features

- Real-time magnitude estimation using GNSS data
- Support for multiple GNSS stations
- High accuracy with mean absolute error ~0.076
- Fast inference time
- Comprehensive visualization tools

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

1. Clone the repository:

```bash
git clone https://github.com/lidwaa/GNSS-Magnitude-Estimator.git
cd GNSS-Magnitude-Estimator
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install additional visualization tools:

```bash
pip install pydot
winget install graphviz  # Windows
# apt-get install graphviz  # Linux
# brew install graphviz     # Mac
```

## Data Structure

The model expects input data in the following format:

- Input shape: (N, 3, 181, 3)

  - N: Number of samples
  - 3: Number of GNSS stations
  - 181: Time window length (seconds)
  - 3: Components (East, North, Up)

- Target shape: (N,)
  - Contains actual earthquake magnitudes

## Usage

1. Prepare your data in the required format and place it in the `data/GNSS_M3S_181/` directory:

   - `xdata.npy`: Input features
   - `ydata.npy`: Target magnitudes

2. Run the model:

```bash
python main.py
```

3. View the results in the `output/model/` directory.

## Model Architecture

The model uses a combination of convolutional and dense layers to process the GNSS time series data:

1. Input Layer: Accepts GNSS displacement data
2. Feature Extraction: Convolutional layers
3. Dense Layers: Magnitude estimation
4. Output: Single neuron for magnitude prediction

## Performance

Current model performance metrics:

- Mean Absolute Error: ~0.076 magnitude units
- Minimum Error: 0.0
- Maximum Error: 0.6
- RMSE: ~0.114

## Visualization

The project includes several visualization tools:

- GNSS component visualization
- Magnitude distribution plots
- Station comparison plots
- Training history visualization
- Prediction performance plots
- Error distribution analysis

To generate visualizations:

```bash
python create_presentation_plots.py
```

## Project Structure

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Project Link: https://github.com/lidwaa/GNSS-Magnitude-Estimator

### GNSS_Magnitude_V1

This is our preliminary DL model based on a convolutional neural network
for magnitude estimation from HR-GNSS data (1 Hz sampling rate).

We have trained the model for three cases:

- Magnitude estimation from 3 stations, 181 seconds, 3 components.
- Magnitude estimation from 7 stations, 181 seconds, 3 components.
- Magnitude estimation from 7 stations, 501 seconds, 3 components.

## Related work:

Claudia Quinteros-Cartaya, Jonas Köhler, Wei Li, Johannes Faber,
Nishtha Srivastava, Exploring a CNN model for earthquake magnitude
estimation using HR-GNSS data, Journal of South American
Earth Sciences, 2024, 104815, ISSN 0895-9811,
https://doi.org/10.1016/j.jsames.2024.104815.

## Getting Started

# Clone the repository

```
git clone https://github.com/srivastavaresearchgroup/GNSS-Magnitude-Estimator
```

# Install dependencies (with python 3.8)

(virtualenv is recommended)

```
pip install -r requirements.txt
```

# Data

The database for each configuration/case is in `./data`.
For example the data folder: `./data/GNSS_M3S_181` contains the data for
3 stations, 181 seconds.

The data is in numpy format, previously selected from the open-access
database published by Lin et al., 2020
(https://doi.org/10.5281/zenodo.4008690).

You can find the information related to the data (ID, Hypocenter,
Magnitude) in the dataframes info_data.csv located in the same folder as
the respective data.

You can use Data_plot.ipynb to plot the waveforms.

Refer to Quinteros et al., 2024 (https://doi.org/10.1016/j.jsames.2024.104815) for
more details about the data configuration.

# Pretrained models and results

The pre-trained models are located in `./trained_models`.
These models are described in Quinteros et al., 2024
(https://doi.org/10.1016/j.jsames.2024.104815).
You can find the respective results in `./predictions`.

# Processing and output

If you want to change the default configuration, you can edit the
variables in `main.py`.

Configuration parameters you should change, depending on your choice:

- Number of stations (nst)
- Time window length (nt)
- Paths/folder names

To split, train and test the data, run:

```
python main.py
```

The outputs will be located in the path that you set.

By default the outputs will be saved in `./tests`, and in the subfolders:

- `./data_inf`: you can find the numpy files with the data index for
  train, validation, and test dataset, associated with the initial
  xdata.npy file.
- `./models`: trained models are saved in this folder
- `./predictions`: the predicted magnitude and error values are saved in
  text files in this folder.
- `./out_log`: the output from logging is saved in this folder.
