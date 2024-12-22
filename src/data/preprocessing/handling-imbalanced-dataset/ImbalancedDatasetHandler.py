from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

class ImbalanceHandler(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.smote = SMOTE(random_state=random_state)
        self._is_fitted = False

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is not None:
            if self._is_fitted:
                return X, y
            else:
                self._is_fitted = True
                X_resampled, y_resampled = self.smote.fit_resample(X, y)
                return X_resampled, y_resampled
        return X
