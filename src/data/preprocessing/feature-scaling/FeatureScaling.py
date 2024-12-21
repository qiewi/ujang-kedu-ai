import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method):
        # Choose scaler based on the method
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Invalid method. Choose 'standard', 'minmax', or 'robust'.")
        self.numeric_columns = None

    def fit(self, X, y=None):
        # Identify numerical columns to scale without id and label
        self.numeric_columns = [
            col for col in X.select_dtypes(include=['float64', 'int64']).columns
            if col not in ['id', 'label']
        ]
        self.scaler.fit(X[self.numeric_columns])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])
        return X

    def save(self, X, current_dir):
        # Define the scaled data path
        scaled_data_path = os.path.join(current_dir, 'scaled_data.csv')
        X.to_csv(scaled_data_path, index=False)
        print(f"Scaled dataset saved to {scaled_data_path}")

# Dataset Path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../../train.csv')

# Load the dataset
df = pd.read_csv(data_path)

# Initialization
scaler = FeatureScaler(method='standard')  # Changeable
scaler.fit(df)
df_scaled = scaler.transform(df)

# Data Saving
scaler.save(df_scaled, current_dir)
