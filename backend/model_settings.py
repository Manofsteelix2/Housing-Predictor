"""One shared rule: every CSV column except target/control is a predictor."""

import json
from pathlib import Path

import pandas as pd


def read_dataset(path):
    data = pd.read_csv(path)
    # Support ordinary CSV quoting and apostrophe-wrapped AHS export codes.
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].map(
            lambda value: value.strip().strip("'") if isinstance(value, str) else value
        ).replace("", float("nan"))
        present = data[column].dropna()
        if pd.to_numeric(present, errors="coerce").notna().all():
            data[column] = pd.to_numeric(data[column])
    return data


def load_config(path, data_path=None):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as file:
        config = json.load(file)
    target = config["target"]
    control = config.get("control_column", "CONTROL")
    if not isinstance(target, str) or not target or target == control:
        raise ValueError("Specify distinct target and control column names.")
    data_path = Path(data_path) if data_path is not None else path.parent.parent / "data" / config["data_file"]
    data = read_dataset(data_path)
    if target not in data:
        raise ValueError(f"Target '{target}' is missing from {data_path.name}.")
    predictors = [name for name in data.columns if name not in (target, control)]
    if not predictors:
        raise ValueError("The CSV must contain at least one predictor.")
    overrides = config.get("feature_types", {})
    if any(kind not in ("numeric", "categorical") for kind in overrides.values()):
        raise ValueError("feature_types values must be numeric or categorical.")
    numeric, categorical = [], []
    for name in predictors:
        inferred = "numeric" if pd.api.types.is_numeric_dtype(data[name]) else "categorical"
        kind = overrides.get(name, inferred)
        (numeric if kind == "numeric" else categorical).append(name)
    config.update(predictors=predictors, numeric_features=numeric, categorical_features=categorical)
    config.setdefault("target_transform", "none")
    if config["target_transform"] not in ("none", "log1p"):
        raise ValueError("target_transform must be none or log1p.")
    return config


def default_feature_schema(config, data):
    """New columns work without hand-authored frontend metadata."""
    schema = {}
    for name in config["predictors"]:
        metadata = {"label": name.replace("_", " ").title(), "input_type": "number", "step": "any"}
        if name in config["categorical_features"]:
            metadata = {"label": metadata["label"], "input_type": "select",
                        "options": data[name].dropna().drop_duplicates().tolist()}
        schema[name] = metadata
    return schema
