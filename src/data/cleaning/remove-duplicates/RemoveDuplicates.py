from sklearn.base import BaseEstimator, TransformerMixin

class DuplicateHandler(BaseEstimator, TransformerMixin):
  def __init__(self, subset=None, keep='first'):
    self.subset = subset
    self.keep = keep

  def fit(self, X, y=None):
    return self

  def fit_transform(self, X, y=None):
    X = X.copy()
    if y is not None:
        unique_idx = ~X.duplicated(subset=self.subset, keep=self.keep)
        X_unique = X[unique_idx].reset_index(drop=True)
        y_unique = y[unique_idx].reset_index(drop=True)
        return X_unique, y_unique
    return X.drop_duplicates(subset=self.subset, keep=self.keep).reset_index(drop=True)

  def transform(self, X, y=None):
    X = X.copy()
    if y is not None:
        return X, y
    return X.drop_duplicates(subset=self.subset, keep=self.keep).reset_index(drop=True)

print(f"Number of rows before removing duplicates: {df.shape[0]}")

duplicate_handler = DuplicateHandler(subset=[col for col in df.columns if col != 'id'])

# Remove duplicates using fit_transform
df_cleaned = duplicate_handler.fit_transform(df)

print(f"Number of rows after removing duplicates: {df_cleaned.shape[0]}")

print(f"Number of duplicates removed: {df.shape[0] - df_cleaned.shape[0]}")