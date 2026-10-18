"""Predictor cleaning shared by training and saved-pipeline predictions."""

import numpy as np
import pandas as pd
from numbers import Real
from sklearn.base import BaseEstimator, TransformerMixin


def categorical_strings(values):
    """Give coded and text categories a common type; leave missing values as NaN."""
    def convert(value):
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            return np.nan
        # Numeric category codes must match whether JSON supplies 1 or 1.0.
        if isinstance(value, Real) and float(value).is_integer():
            return str(int(value))
        return str(value)

    frame = pd.DataFrame(values)
    return frame.apply(lambda column: column.map(convert)).to_numpy(dtype=object)


class HousingCodeCleaner(TransformerMixin, BaseEstimator):
    """Replace negative predictor codes and invalid sizes with NaN; retain rows."""

    def __init__(self, lot_size_categories=False):
        self.lot_size_categories = lot_size_categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = X.copy()
        # All negative numeric values are survey missing codes, including codes
        # in categorical/object columns. Preserve ordinary text category values.
        for column in cleaned.columns:
            numeric_values = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = cleaned[column].mask(numeric_values.lt(0), np.nan)
        for column in ("UNITSIZE", "LOTSIZE"):
            if column not in cleaned.columns:
                continue
            values = pd.to_numeric(cleaned[column], errors="coerce")
            invalid = (values < 0) | ~np.isfinite(values)
            if column == "LOTSIZE" and self.lot_size_categories:
                invalid = invalid | ~values.isin(range(1, 8))
            cleaned[column] = values.mask(invalid, np.nan)
        return cleaned
