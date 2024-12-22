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
        numeric_cols = X.select_dtypes(include=[np.number]).columns

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
        if not self.bounds:
            raise ValueError("The model is not fitted yet. Please call 'fit' first.")
        
        X = X.copy()

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
        return X

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../handling-missing-data/cleaned_data.csv')

df = pd.read_csv(data_path)

df = df.replace([np.inf, -np.inf], np.nan).dropna()

handler = OutlierHandler(method='iqr', multiplier=1.5, strategy='clip')
handler.fit(df)
df_no_outliers = handler.transform(df)

cleaned_data_path = os.path.join(current_dir, 'cleaned_outliers.csv')
df_no_outliers.to_csv(cleaned_data_path, index=False)

print("Data cleaned and saved to:", cleaned_data_path)
