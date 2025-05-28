import numpy as np
import pandas as pd
import tensorflow as tf
import os
import argparse
from typing import Tuple, Dict

class GNSSMagnitudePredictor:
    def __init__(self, model_path: str = 'output/model/model.h5'):
        """Initialize the GNSS Magnitude Predictor.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = tf.keras.models.load_model(model_path)
        
    def csv_to_numpy(self, csv_file: str) -> np.ndarray:
        """Convert CSV format to numpy array.
        
        Args:
            csv_file: Path to input CSV file
            
        Returns:
            numpy array of shape (3, 181, 3) containing GNSS data
        """
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Verify data format
        required_columns = [
            'time_second',
            'station1_east', 'station1_north', 'station1_up',
            'station2_east', 'station2_north', 'station2_up',
            'station3_east', 'station3_north', 'station3_up'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
            
        if len(df) != 181:
            raise ValueError(f"CSV must contain exactly 181 rows (found {len(df)})")
        
        # Create empty numpy array
        data = np.zeros((3, 181, 3))
        
        # Fill the array with data from CSV
        for second in range(181):
            row = df.iloc[second]
            
            # Station 1
            data[0, second, 0] = row['station1_east']
            data[0, second, 1] = row['station1_north']
            data[0, second, 2] = row['station1_up']
            
            # Station 2
            data[1, second, 0] = row['station2_east']
            data[1, second, 1] = row['station2_north']
            data[1, second, 2] = row['station2_up']
            
            # Station 3
            data[2, second, 0] = row['station3_east']
            data[2, second, 1] = row['station3_north']
            data[2, second, 2] = row['station3_up']
        
        return data
    
    def predict(self, data: np.ndarray) -> float:
        """Predict earthquake magnitude from GNSS data.
        
        Args:
            data: numpy array of shape (3, 181, 3)
            
        Returns:
            Predicted earthquake magnitude
        """
        # Add batch dimension
        x = np.expand_dims(data, axis=0)
        
        # Make prediction
        prediction = self.model.predict(x, verbose=0)
        
        return float(prediction[0])
    
    def predict_from_csv(self, csv_file: str) -> Dict:
        """Predict earthquake magnitude from CSV file.
        
        Args:
            csv_file: Path to input CSV file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Convert CSV to numpy array
            data = self.csv_to_numpy(csv_file)
            
            # Make prediction
            magnitude = self.predict(data)
            
            return {
                'success': True,
                'predicted_magnitude': magnitude,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'predicted_magnitude': None,
                'error': str(e)
            }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict earthquake magnitude from GNSS data CSV')
    parser.add_argument('csv_file', help='Path to input CSV file')
    parser.add_argument('--model', default='output/model/model.h5', help='Path to model file')
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Initialize predictor
    predictor = GNSSMagnitudePredictor(args.model)
    
    # Make prediction
    result = predictor.predict_from_csv(args.csv_file)
    
    if result['success']:
        print(f"\nPredicted Earthquake Magnitude: {result['predicted_magnitude']:.2f}")
    else:
        print(f"\nError: {result['error']}")
        
if __name__ == '__main__':
    main() 