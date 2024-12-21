import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import os

class MissingDataHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_imputer = {}
        self.categorical_imputer = {}

    def fit(self, X, y=None):
        # Categorical and numerical features
        categorical_features = ['state', 'service']
        numerical_features = [
            'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
            'sload', 'dload', 'spkts', 'dpkts', 'id'
        ]

        # Compute the median for numerical features
        for col in numerical_features:
            if col in X.columns:
                self.numeric_imputer[col] = X[col].median()

        # Compute the mode for categorical features
        for col in categorical_features:
            if col in X.columns:
                self.categorical_imputer[col] = X[col].mode()[0]

        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Impute numerical features
        for col, value in self.numeric_imputer.items():
            if col in X.columns:
                X[col].fillna(value, inplace=True)

        # Impute categorical features
        for col, value in self.categorical_imputer.items():
            if col in X.columns:
                X[col].fillna(value, inplace=True)

        return X

# Dataset Path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../../basic_features_train.csv')

# Load the dataset
df = pd.read_csv(data_path)

# Initialization
handler = MissingDataHandler()
handler.fit(df)
df_cleaned = handler.transform(df)

# Save cleaned data
cleaned_data_path = os.path.join(current_dir, 'cleaned_data.csv')
df_cleaned.to_csv(cleaned_data_path, index=False)
