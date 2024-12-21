import pandas as pd
import numpy as np
import os

class OutlierHandler:
    def __init__(self, method='iqr', multiplier=1.5, strategy='clip', transform=None):
        """
        method: Outlier detection method ('iqr' supported).
        multiplier: Multiplier for IQR to calculate bounds.
        strategy: Strategy to handle outliers ('clip', 'mean', 'median').
        transform: Transformation to apply ('log', 'sqrt', or None).
        """
        self.method = method
        self.multiplier = multiplier
        self.strategy = strategy
        self.transform = transform
        self.bounds = {}

    def fit(self, X):
        # Identify numeric columns in the dataset
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # For each numeric column, calculate Q1, Q3, and IQR
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate lower and upper bounds based on IQR and multiplier
            self.bounds[col] = {
                'lower': Q1 - self.multiplier * IQR,
                'upper': Q3 + self.multiplier * IQR
            }
        print("Bounds calculated for numeric columns:", self.bounds)
        return self

    def transform(self, X):
        if not self.bounds:
            raise ValueError("Bounds are not calculated. Please call 'fit' before 'transform'.")
        
        # Create a copy of the dataset to avoid modifying the original data
        X = X.copy()

        # Apply transformations to reduce skewness if specified
        if self.transform == 'log':
            X = X.applymap(lambda x: np.log1p(x) if np.issubdtype(type(x), np.number) and x > 0 else x)
        elif self.transform == 'sqrt':
            X = X.applymap(lambda x: np.sqrt(x) if np.issubdtype(type(x), np.number) and x > 0 else x)

        # Handle outliers using the specified strategy
        for col, bounds in self.bounds.items():
            if col in X.columns:
                if self.strategy == 'clip':
                    X[col] = np.clip(X[col], bounds['lower'], bounds['upper'])
                elif self.strategy == 'mean':
                    mean_value = X[col].mean()
                    X[col] = X[col].where(
                        (X[col] >= bounds['lower']) & (X[col] <= bounds['upper']),
                        mean_value
                    )
                elif self.strategy == 'median':
                    median_value = X[col].median()
                    X[col] = X[col].where(
                        (X[col] >= bounds['lower']) & (X[col] <= bounds['upper']),
                        median_value
                    )
        print("Outliers handled for numeric columns.")
        return X

# Dataset Path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../../train.csv')

# Load the dataset
df = pd.read_csv(data_path)

# Initialize and apply the OutlierHandler
handler = OutlierHandler(method='iqr', multiplier=1.5, strategy='mean')  # Correct parameter name 'strategy'
handler.fit(df)  # Fit to calculate bounds
df_no_outliers = handler.transform(df)  # Transform to handle outliers

# Save the cleaned data
cleaned_data_path = os.path.join(current_dir, 'cleaned_outliers.csv')
df_no_outliers.to_csv(cleaned_data_path, index=False)

print("Data cleaned and saved to:", cleaned_data_path)
