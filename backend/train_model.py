"""Train and save a configurable housing-price regression pipeline."""

import json
import hashlib
import math
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # Save plots without requiring a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from housing_preprocessing import HousingCodeCleaner
from model_settings import load_config, read_dataset


BACKEND_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BACKEND_DIR / "model_config.json"
DATA_PATH = BACKEND_DIR.parent / "data" / "housingB.csv"
MODEL_PATH = BACKEND_DIR / "model.pkl"
DIAGNOSTICS_DIR = BACKEND_DIR / "diagnostics"


def load_data(path, config):
    # Loading data: validate every configured column before selecting predictors.
    try:
        data = read_dataset(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Dataset is empty: {path}") from error
    target = config["target"]
    features = config["predictors"]
    missing_columns = [name for name in [target] + features if name not in data.columns]
    if missing_columns:
        raise ValueError("Configured columns missing from housingB.csv: " + ", ".join(missing_columns))
    # Inspect original codes before cleaning or filtering any rows.
    for column in ("UNITSIZE", "LOTSIZE"):
        if column in data.columns:
            counts = data[column].value_counts(dropna=False).sort_index()
            print(f"\n{column}: unique values and frequency counts (original CSV)")
            print(counts.to_string())
    if "LOTSIZE" in features:
        lot_values = pd.to_numeric(data["LOTSIZE"], errors="coerce")
        valid_lots = lot_values[lot_values.ge(0) & np.isfinite(lot_values)]
        if not valid_lots.empty and valid_lots.isin(range(1, 8)).all():
            print("LOTSIZE uses codes 1–7; model these as categories, not numeric lot area.")
            if "LOTSIZE" not in config["categorical_features"]:
                raise ValueError("Set feature_types.LOTSIZE to categorical in model_config.json for 1–7 codes.")
    if "UNITSIZE" in features and "UNITSIZE" not in config["numeric_features"]:
        raise ValueError("Set feature_types.UNITSIZE to numeric in model_config.json.")
    data = data[[target] + features].copy()
    data = data.replace(r"^\s*$", np.nan, regex=True)
    # Housing prices must be positive; negative AHS target codes are not prices.
    data[target] = pd.to_numeric(data[target], errors="raise")
    invalid_target = data[target].notna() & (data[target] <= 0)
    print(f"Excluded {invalid_target.sum():,} rows with nonpositive target values/special codes.")
    data = data.loc[~invalid_target].copy()
    # Replace special predictor codes, retaining rows for train-only imputation.
    cleaner = HousingCodeCleaner("LOTSIZE" in config["categorical_features"])
    cleaned_features = cleaner.transform(data[features])
    for column in ("UNITSIZE", "LOTSIZE"):
        if column in features:
            newly_missing = data[column].notna() & cleaned_features[column].isna()
            print(f"{column}: replaced {newly_missing.sum():,} invalid values with missing; rows retained.")
    data[features] = cleaned_features
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
    if config.get("target_transform", "none") == "log1p" and (data[target] <= -1).any():
        raise ValueError(f"Target '{target}' must be greater than -1 for log1p transformation.")
    return data[features], data[target]


def print_dataset_summary(features, target):
    # Describe the full modeling dataset after target filtering and validation.
    print("\nDataset summary (after target filtering)")
    print(f"Rows used for modeling: {len(target):,}")
    print(f"Number of predictors: {features.shape[1]}")
    statistics = {
        "mean": target.mean(),
        "median": target.median(),
        "minimum": target.min(),
        "maximum": target.max(),
        "standard deviation (sample)": target.std(),
    }
    for name, value in statistics.items():
        print(f"{target.name} {name}: {value:,.4f}")
    print(f"\n{target.name} percentiles")
    for percentile in [25, 50, 75, 90, 95, 99]:
        print(f"{percentile}th percentile: {target.quantile(percentile / 100):,.4f}")


def save_diagnostic_plots(target, actual, predicted, diagnostics_dir, model_name=""):
    # Distribution plots use all modeling rows; error plots use only test rows.
    diagnostics_dir = Path(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    target_name = target.name
    prefix = f"{model_name}_" if model_name else ""
    label = f"{model_name.title()} model: " if model_name else ""

    def save_plot(fig, filename):
        fig.tight_layout()
        fig.savefig(diagnostics_dir / filename, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(target, bins=50, edgecolor="white")
    ax.set(title=f"{target_name} distribution (modeling rows)",
           xlabel=target_name, ylabel="Number of properties")
    save_plot(fig, f"{target_name.lower()}_histogram.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(target)
    ax.set_xticks([1], [target_name])
    ax.set(title=f"{target_name} boxplot (modeling rows)", ylabel=target_name)
    save_plot(fig, f"{target_name.lower()}_boxplot.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.3, s=12)
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    ax.plot([lower, upper], [lower, upper], "r--", label="Perfect prediction (y = x)")
    ax.set(xlim=(lower, upper), ylim=(lower, upper),
           title=f"{label}actual vs predicted {target_name} (test rows)",
           xlabel=f"Actual {target_name}", ylabel=f"Predicted {target_name}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    save_plot(fig, f"{prefix}actual_vs_predicted.png")

    residuals = actual.to_numpy() - predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", label="Zero error")
    ax.set(title=f"{label}{target_name} residual distribution (test rows)",
           xlabel=f"Residual (actual - predicted {target_name})",
           ylabel="Number of properties")
    ax.legend()
    save_plot(fig, f"{prefix}residual_distribution.png")
    print(f"Saved diagnostic plots to: {diagnostics_dir}")


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
    regressor = LinearRegression()
    if config.get("target_transform", "none") == "log1p":
        # Fit on log1p prices; predict on the log scale, then invert with expm1.
        # Keeping the inverse inside the pipeline also preserves dollar outputs
        # after loading model.pkl, without requiring extra prediction logic.
        regressor = TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1
        )
    return Pipeline([
        # Persist code cleaning so raw inputs receive the same treatment later.
        ("code_cleaner", HousingCodeCleaner("LOTSIZE" in config["categorical_features"])),
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def train_model(data_path=None, config_path=CONFIG_PATH, model_path=MODEL_PATH,
                diagnostics_dir=DIAGNOSTICS_DIR):
    config = load_config(config_path, data_path)
    data_path = Path(data_path) if data_path is not None else BACKEND_DIR.parent / "data" / config.get("data_file", DATA_PATH.name)
    features, target = load_data(data_path, config)
    # Both models need the same eligible rows, even when saving the raw model.
    if (target <= -1).any():
        raise ValueError("Raw/log comparison requires targets greater than -1 for log1p.")
    print_dataset_summary(features, target)

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

    models, results = {}, {}
    print("\nRaw/log comparison: identical predictors and train/test rows")
    print(f"Training rows: {len(x_train):,} | Testing rows: {len(x_test):,}")
    for name, transform in (("raw", "none"), ("log", "log1p")):
        # Model training: only the target transformation differs between models.
        model = build_pipeline({**config, "target_transform": transform})
        model.fit(x_train, y_train)
        # Evaluation: log-model predict() applies expm1 before all metrics/plots.
        predictions = model.predict(x_test)
        if not np.isfinite(predictions).all():
            raise ValueError(f"{name} predictions are nonfinite; model was not saved.")
        absolute_errors = np.abs(y_test.to_numpy() - predictions)
        metrics = {
            "R²": r2_score(y_test, predictions),
            "RMSE": math.sqrt(mean_squared_error(y_test, predictions)),
            "MAE": mean_absolute_error(y_test, predictions),
            "Median Absolute Error": np.median(absolute_errors),
            "Max Absolute Error": absolute_errors.max(),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"{name} evaluation produced nonfinite metrics; model was not saved.")
        models[name], results[name] = model, metrics
        save_diagnostic_plots(target, y_test, predictions, diagnostics_dir, name)

    # Console comparison: all errors are dollars, and R² uses dollar predictions.
    print("\nModel comparison (test rows; original dollar scale)")
    columns = list(results["raw"])
    print("Model | " + " | ".join(columns))
    for name, metrics in results.items():
        print(name.title() + " | " + " | ".join(f"{metrics[key]:,.4f}" for key in columns))

    # Model saving: persist the ENTIRE fitted pipeline only after successful evaluation.
    selected = "log" if config["target_transform"] == "log1p" else "raw"
    model, metrics = models[selected], results[selected]
    # Keep training provenance inside the artifact so stale models can be detected.
    model.training_metadata_ = {
        "data_file": data_path.name,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "config": config,
    }
    joblib.dump(model, model_path)
    print(f"Saved configured {selected} pipeline to: {model_path}")

    print("\nMetric differences (same test rows)")
    for metric, direction in (("R²", "higher"), ("RMSE", "lower"), ("MAE", "lower")):
        raw_value, log_value = results["raw"][metric], results["log"][metric]
        difference = abs(raw_value - log_value)
        if raw_value == log_value:
            print(f"{metric}: tied (difference 0).")
        else:
            raw_wins = raw_value > log_value if direction == "higher" else raw_value < log_value
            winner = "Raw" if raw_wins else "Log"
            print(f"{winner} has {direction} {metric} by {difference:,.4f}.")
    print("These are metric-specific comparisons; neither model is declared universally better.")
    return model, metrics


if __name__ == "__main__":
    try:
        train_model()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        sys.exit(1)
