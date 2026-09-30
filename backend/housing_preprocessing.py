"""Predictor cleaning shared by training and saved-pipeline predictions."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HousingCodeCleaner(TransformerMixin, BaseEstimator):
    """Replace invalid size values with NaN without removing any rows."""

    def __init__(self, lot_size_categories=False):
        self.lot_size_categories = lot_size_categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = X.copy()
        for column in ("UNITSIZE", "LOTSIZE"):
            if column not in cleaned.columns:
                continue
            values = pd.to_numeric(cleaned[column], errors="coerce")
            invalid = (values < 0) | ~np.isfinite(values)
            if column == "LOTSIZE" and self.lot_size_categories:
                invalid = invalid | ~values.isin(range(1, 8))
            cleaned[column] = values.mask(invalid, np.nan)
        return cleaned
