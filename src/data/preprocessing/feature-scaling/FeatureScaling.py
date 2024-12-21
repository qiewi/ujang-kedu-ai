import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method): 
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
        # Numerical columns for scaling
        self.numeric_columns = [
            'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
            'sload', 'dload', 'spkts', 'dpkts', 'id'
        ]
        self.numeric_columns = [col for col in self.numeric_columns if col in X.columns]
        self.scaler.fit(X[self.numeric_columns])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])
        return X

# Dataset Path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../../basic_features_train.csv')

df = pd.read_csv(data_path)

# Initialization
scaler = FeatureScaler(method='standard')  # Changeable
scaler.fit(df)
df_scaled = scaler.transform(df)

# Save scaled data
scaled_data_path = os.path.join(current_dir, 'scaled_data.csv')
df_scaled.to_csv(scaled_data_path, index=False)

