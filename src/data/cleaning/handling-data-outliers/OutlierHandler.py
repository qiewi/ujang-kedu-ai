import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', multiplier=1.5, strategy='clip'):
        """
        method: Outlier detection method ('iqr' supported currently).
        multiplier: Multiplier for IQR to calculate bounds (used only with 'iqr' method).
        strategy: Strategy to handle outliers ('clip', 'mean', 'median').
        """
        self.method = method
        self.multiplier = multiplier
        self.strategy = strategy
        self.bounds = {}

    def fit(self, X, y=None):
        # Automatically detect numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Calculate bounds for each numeric column using IQR method
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = {
                'lower': Q1 - self.multiplier * IQR,
                'upper': Q3 + self.multiplier * IQR
            }
        print(f"Bounds calculated for columns: {self.bounds}")
        return self

    def transform(self, X, y=None):
        # Ensure that bounds are calculated before transforming
        if not self.bounds:
            raise ValueError("The model is not fitted yet. Please call 'fit' first.")
        
        X = X.copy()

        # Handle outliers based on the calculated bounds
        for col, bounds in self.bounds.items():
            if col in X.columns:
                if self.strategy == 'clip':
                    # Clipping the values between lower and upper bounds
                    X[col] = np.clip(X[col], bounds['lower'], bounds['upper'])
                elif self.strategy == 'mean':
                    # Replace outliers with the mean of the column
                    mean_value = X[col].mean()
                    X[col] = X[col].where(
                        (X[col] >= bounds['lower']) & (X[col] <= bounds['upper']),
                        mean_value
                    )
                elif self.strategy == 'median':
                    # Replace outliers with the median of the column
                    median_value = X[col].median()
                    X[col] = X[col].where(
                        (X[col] >= bounds['lower']) & (X[col] <= bounds['upper']),
                        median_value
                    )
        return X

# Dataset Path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../handling-missing-data/cleaned_data.csv')

# Load the dataset
df = pd.read_csv(data_path)

# Clean up any NaN or infinite values before proceeding
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Initialize and apply the OutlierHandler
handler = OutlierHandler(method='iqr', multiplier=1.5, strategy='clip')  # Renamed parameter
handler.fit(df)  # Fit to calculate bounds
df_no_outliers = handler.transform(df)  # Transform to handle outliers

# Save the cleaned data
cleaned_data_path = os.path.join(current_dir, 'cleaned_outliers.csv')
df_no_outliers.to_csv(cleaned_data_path, index=False)

print("Data cleaned and saved to:", cleaned_data_path)
