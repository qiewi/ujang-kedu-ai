import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MissingDataHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_imputer = {}
        self.categorical_imputer = {}

    def fit(self, X, y=None):
        # Spcify categorical and numerical features
        categorical_features = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
        numerical_features = [
            col for col in X.columns if col not in categorical_features + ['id', 'label']
        ]

        # Compute imputers
        for col in numerical_features:
            if col in X.columns:
                self.numeric_imputer[col] = X[col].median()

        for col in categorical_features:
            if col in X.columns:
                self.categorical_imputer[col] = X[col].mode()[0] if not X[col].mode().empty else None

        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Impute missing numerical values
        for col, value in self.numeric_imputer.items():
            if col in X.columns:
                X[col].fillna(value, inplace=True)
        
        # Impute missing categorical values
        for col, value in self.categorical_imputer.items():
            if col in X.columns and value is not None:
                X[col].fillna(value, inplace=True)
        return X

    def save(self, X, current_dir):
        # Define the cleaned data path
        cleaned_data_path = os.path.join(current_dir, 'cleaned_data.csv')
        X.to_csv(cleaned_data_path, index=False)
        print(f"Dataset saved to {cleaned_data_path}")

# Dataset Path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../../train.csv')

# Load the dataset
df = pd.read_csv(data_path)

# Initialization
handler = MissingDataHandler()
handler.fit(df)
df_cleaned = handler.transform(df)

# Data Saving
handler.save(df_cleaned, current_dir)
