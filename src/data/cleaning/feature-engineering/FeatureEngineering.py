import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PhishingFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def calculate_capital_ratio(self, url):
        url = str(url)
        capital_count = sum(1 for char in url if char.isupper())
        return capital_count / (len(url) + 1)

    def check_phishing_keywords(self, url):
        phishing_keywords = ['login', 'secure', 'account', 'bank', 'verify']
        url = str(url).lower()
        for keyword in phishing_keywords:
            if keyword in url:
                return 1
        return 0

    def count_url_segments(self, url):
        url = str(url)
        return url.count('/')

    def calculate_special_char_ratio(self, url):
        special_chars = set('!@#$%^&*()[]{}|\\:;"\'<>,.?/~`-=_+')
        url = str(url)
        special_char_count = sum(1 for char in url if char in special_chars)
        return special_char_count / (len(url) + 1)

    def transform(self, X, y=None):
        X = X.copy()

        if 'URL' in X.columns:
            X['capital_ratio'] = X['URL'].apply(self.calculate_capital_ratio)

        if 'URL' in X.columns:
            X['contains_phishing_keywords'] = X['URL'].apply(self.check_phishing_keywords)

        if 'URL' in X.columns:
            X['url_segment_count'] = X['URL'].apply(self.count_url_segments)

        if 'URL' in X.columns:
            X['special_char_ratio'] = X['URL'].apply(self.calculate_special_char_ratio)

        if 'TLDLegitimateProb' in X.columns:
            X['TLDLegitimateProb'] = X['TLDLegitimateProb'].fillna(0)

        if 'NoOfSubDomain' in X.columns:
            X['NoOfSubDomain'] = X['NoOfSubDomain'].fillna(0)

        return X
