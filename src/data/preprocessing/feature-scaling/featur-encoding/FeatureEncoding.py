import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import os

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Categorical columns are predefined within the class.
        """
        self.categorical_columns = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
        self.label_encoders = {}

    def fit(self, X, y=None):
        # Initialize LabelEncoders for each categorical column
        for col in self.categorical_columns:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(X[col].astype(str))
        return self

    def transform(self, X):
        X = X.copy()

        # Apply Label Encoding to each categorical column
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))

        return X
    
    def save(self, X, current_dir):
        # Define the scaled data path
        scaled_data_path = os.path.join(current_dir, 'scaled_data.csv')
        X.to_csv(scaled_data_path, index=False)
        print(f"Scaled dataset saved to {scaled_data_path}")
    
# Dataset Path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../../../train.csv')

# Load the dataset
df = pd.read_csv(data_path)

# Initialize the FeatureEncoder
encoder = FeatureEncoder()

# Fit the encoder on the data
encoder.fit(df)

# Transform the data using the fitted encoder
encoded_data = encoder.transform(df)

# Save the encoded dataset
encoder.save(encoded_data, current_dir)

# Print the encoded dataset
print("Encoded Data:")
print(encoded_data)