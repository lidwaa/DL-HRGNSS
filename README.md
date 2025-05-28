# GNSS-based Earthquake Magnitude Prediction

This project provides a tool for predicting earthquake magnitudes using GNSS (Global Navigation Satellite System) displacement data.

## Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Pandas

Install dependencies:

```bash
pip install tensorflow numpy pandas
```

## Input Data Format

The script expects a CSV file with the following format:

1. Must contain exactly 181 rows (representing 0-180 seconds)
2. Must have the following columns:
   - time_second: Time in seconds (0-180)
   - station1_east: East displacement for station 1 (meters)
   - station1_north: North displacement for station 1 (meters)
   - station1_up: Up displacement for station 1 (meters)
   - station2_east: East displacement for station 2 (meters)
   - station2_north: North displacement for station 2 (meters)
   - station2_up: Up displacement for station 2 (meters)
   - station3_east: East displacement for station 3 (meters)
   - station3_north: North displacement for station 3 (meters)
   - station3_up: Up displacement for station 3 (meters)

Example of CSV format:

```csv
time_second,station1_east,station1_north,station1_up,station2_east,station2_north,station2_up,station3_east,station3_north,station3_up
0,0.001,-0.002,0.003,0.002,0.001,-0.001,0.000,0.001,0.002
1,0.002,-0.003,0.004,0.003,0.002,-0.002,0.001,0.002,0.003
...
```

## Usage

To predict the magnitude for a single earthquake event:

```bash
python predict_magnitude_from_csv.py path/to/your/data.csv
```

Optional arguments:

- `--model`: Path to the model file (default: 'output/model/model.h5')

Example:

```bash
python predict_magnitude_from_csv.py gnss_data.csv --model output/model/model.h5
```

## Output

The script will output either:

- The predicted earthquake magnitude (if successful)
- An error message explaining what went wrong (if unsuccessful)

Example output:

```
Predicted Earthquake Magnitude: 6.75
```

## Error Handling

The script performs several validations:

1. Checks if the input CSV file exists
2. Verifies the CSV has the correct number of rows (181)
3. Validates that all required columns are present
4. Ensures the model file exists and can be loaded

If any of these checks fail, an appropriate error message will be displayed.

## Model Information

The model expects GNSS displacement data with the following dimensions:

- 3 GNSS stations
- 181 seconds of data (0-180 seconds)
- 3 components per station (East, North, Up)

The model has been trained on a dataset of earthquakes with magnitudes ranging from approximately 5.0 to 7.5.
