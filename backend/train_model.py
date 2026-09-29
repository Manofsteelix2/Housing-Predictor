"""Train and save a configurable housing-price regression pipeline."""

import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BACKEND_DIR / "model_config.json"
DATA_PATH = BACKEND_DIR.parent / "data" / "housing.csv"
MODEL_PATH = BACKEND_DIR / "model.pkl"


def load_config(path):
    # Loading configuration: predictor changes belong in JSON, not this script.
    with Path(path).open(encoding="utf-8-sig") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a JSON object.")
    target = config.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Configuration needs a nonempty 'target' column name.")
    for key in ("numeric_features", "categorical_features"):
        columns = config.get(key)
        if not isinstance(columns, list) or any(
            not isinstance(column, str) or not column.strip() for column in columns
        ):
            raise ValueError(f"'{key}' must be a list of nonempty column names.")
    features = config["numeric_features"] + config["categorical_features"]
    if not features:
        raise ValueError("Configure at least one predictor.")
    if len(features) != len(set(features)):
        raise ValueError("Predictor names must be unique across both feature lists.")
    if target in features:
        raise ValueError("The target cannot also be a predictor (data leakage).")
    return config


def load_data(path, config):
    # Loading data: validate every configured column before selecting predictors.
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Dataset is empty: {path}") from error
    target = config["target"]
    features = config["numeric_features"] + config["categorical_features"]
    missing_columns = [name for name in [target] + features if name not in data.columns]
    if missing_columns:
        raise ValueError("Configured columns missing from housing.csv: " + ", ".join(missing_columns))
    data = data[[target] + features].copy()
    data = data.replace(r"^\s*$", np.nan, regex=True)
    removed = int(data[target].isna().sum())
    data = data.dropna(subset=[target])
    print(f"Removed {removed:,} rows with missing target '{target}'.")
    if len(data) < 6:
        raise ValueError("Need at least 6 rows with a target for an 80/20 split and R² evaluation.")
    for column in [target] + config["numeric_features"]:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Column '{column}' contains nonnumeric values; correct them before training.") from error
        if np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Column '{column}' contains infinite values; correct them before training.")
    negative_columns = [
        column for column in [target] + config["numeric_features"]
        if (data[column] < 0).any()
    ]
    if negative_columns:
        print("Data note: negative values are retained in " + ", ".join(negative_columns)
              + ". Check the dataset codebook for special missing-value codes.")
    return data[features], data[target]


def build_pipeline(config):
    # Preprocessing: fit imputers and encoders only on the training partition.
    # Keep empty features so an all-missing predictor is never silently dropped.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, config["numeric_features"]),
        ("categorical", categorical_pipeline, config["categorical_features"]),
    ], remainder="drop")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])


def train_model(data_path=DATA_PATH, config_path=CONFIG_PATH, model_path=MODEL_PATH):
    config = load_config(config_path)
    features, target = load_data(data_path, config)

    # Train/test split: fixed randomness makes this baseline reproducible.
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    if y_test.nunique() < 2:
        raise ValueError("Test target is constant; R² is not meaningful. Supply more varied data.")
    empty_features = x_train.columns[x_train.isna().all()].tolist()
    if empty_features:
        print("Data note: all-missing training predictors retained with a zero fallback: "
              + ", ".join(empty_features))

    # Model training: preprocessing and regression are fitted together.
    model = build_pipeline(config)
    model.fit(x_train, y_train)

    # Evaluation: holdout rows never participate in fitting preprocessing or model.
    predictions = model.predict(x_test)
    metrics = {
        "R²": r2_score(y_test, predictions),
        "RMSE": math.sqrt(mean_squared_error(y_test, predictions)),
        "MAE": mean_absolute_error(y_test, predictions),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation produced nonfinite metrics; model was not saved.")
    print(f"\nLinearRegression evaluation (target: {config['target']})")
    print(f"Training rows: {len(x_train):,} | Testing rows: {len(x_test):,}")
    for name, value in metrics.items():
        print(f"{name}: {value:,.4f}")
    print("RMSE and MAE are expressed in the target's units.")

    # Model saving: persist the ENTIRE fitted pipeline only after successful evaluation.
    joblib.dump(model, model_path)
    print(f"Saved pipeline to: {model_path}")
    return model, metrics


if __name__ == "__main__":
    try:
        train_model()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        sys.exit(1)
