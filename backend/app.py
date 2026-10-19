"""Flask API for the saved housing regression pipeline."""

import json
import hashlib
import math
import os
from numbers import Real
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from model_settings import load_config, read_dataset, default_feature_schema, fitted_feature_config
from werkzeug.exceptions import BadRequest, UnsupportedMediaType


BACKEND_DIR = Path(__file__).resolve().parent


def create_app():
    app = Flask(__name__)
    app.config["DEBUG"] = False
    # CORS uses origins only: a GitHub Pages project path is not part of its origin.
    origins = os.environ.get("CORS_ORIGINS", "https://manofsteelix2.github.io,http://127.0.0.1:8000,http://localhost:8000")
    CORS(app, origins=[origin.strip() for origin in origins.split(",") if origin.strip()])

    # Load configuration and the complete fitted pipeline once, at startup.
    config = load_config(BACKEND_DIR / "model_config.json")
    model_path = BACKEND_DIR / "model.pkl"
    model = joblib.load(model_path)
    metadata = getattr(model, "training_metadata_", {})
    interval = metadata.get("prediction_interval")
    interval_error = "Prediction-interval calibration is missing or invalid. Retrain the model with backend/train_model.py to generate a 95% prediction interval."
    if isinstance(interval, dict):
        quantile = interval.get("conformal_residual_quantile")
        sample_size = interval.get("calibration_sample_size")
        if (interval.get("method") == "split_conformal_absolute_residual"
                and interval.get("coverage_level") == 0.95
                and isinstance(sample_size, int) and not isinstance(sample_size, bool) and sample_size >= 19
                and isinstance(quantile, Real) and not isinstance(quantile, bool)
                and math.isfinite(quantile) and quantile >= 0):
            interval_error = None
    if metadata.get("config") != config:
        raise ValueError("Dataset-derived settings differ from the saved model. Retrain before starting the API.")
    # Advertise the scope of the loaded artifact, not unsaved configuration edits.
    target_range = {key: metadata.get(key) for key in ("target_min", "target_max")}
    fitted_config = fitted_feature_config(config, metadata.get("final_predictors", config["predictors"]))
    numeric = fitted_config["numeric_features"]
    categorical = fitted_config["categorical_features"]
    if any(not isinstance(group, list) or any(
        not isinstance(name, str) or not name.strip() for name in group
    ) for group in (numeric, categorical)):
        raise ValueError("Feature configuration must contain lists of column names.")
    predictors = fitted_config["predictors"]
    excluded = [name for name in config["predictors"] if name not in predictors]
    if not predictors or len(predictors) != len(set(predictors)):
        raise ValueError("Configure at least one predictor, with no duplicate names.")
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    preprocessing = model.named_steps["preprocessor"]
    if list(preprocessing.feature_names_in_) != predictors:
        raise ValueError("Configured predictors differ from model.pkl. Retrain before starting the API.")
    for name, columns in (("numeric", numeric), ("categorical", categorical)):
        trained_columns = next(cols for key, _, cols in preprocessing.transformers_ if key == name)
        if list(trained_columns) != columns:
            raise ValueError("Configured feature types differ from model.pkl. Retrain the model.")

    # Infer category value types from the fitted encoder (e.g. numeric lot codes
    # versus future neighborhood names), without hardcoding predictor names.
    numeric_categories = set(metadata.get("numeric_categorical_features", []))
    if categorical and "numeric_categorical_features" not in metadata:
        encoder = preprocessing.named_transformers_["categorical"].named_steps["encoder"]
        for name, categories in zip(categorical, encoder.categories_):
            if all(isinstance(value, Real) for value in categories):
                numeric_categories.add(name)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_sha256": model_sha256})

    @app.get("/model-info")
    def model_info():
        # Re-read display metadata, but never advertise fields that differ from
        # the loaded model. Predictor changes require retraining and a restart.
        try:
            current = load_config(BACKEND_DIR / "model_config.json")
            if current != config:
                raise ValueError("Predictor configuration changed. Retrain and restart the API.")
            data = read_dataset(BACKEND_DIR.parent / "data" / current["data_file"],
                                current.get("derived_binary_features"))
            schema = default_feature_schema(fitted_config, data)
            schema_path = BACKEND_DIR / "feature_schema.json"
            if schema_path.exists():
                with schema_path.open(encoding="utf-8-sig") as file:
                    overrides = json.load(file)
                if not isinstance(overrides, dict):
                    raise ValueError("feature_schema.json must contain an object.")
                for name in schema:
                    if name in overrides:
                        schema[name].update(overrides[name])
            features = []
            for name in predictors:
                metadata = schema.get(name)
                if not isinstance(metadata, dict):
                    raise ValueError(f"Missing frontend metadata for configured predictor '{name}'.")
                if not isinstance(metadata.get("label"), str) or not metadata["label"].strip():
                    raise ValueError(f"Missing display label for '{name}'.")
                kind = metadata.get("input_type")
                if not isinstance(metadata.get("required", False), bool):
                    raise ValueError(f"required for '{name}' must be a boolean.")
                if kind not in ("number", "select"):
                    raise ValueError(f"Unsupported input_type for '{name}'; use number or select.")
                value_type = "number" if name in numeric or name in numeric_categories else "string"
                if kind == "number" and value_type != "number":
                    raise ValueError(f"Text predictor '{name}' requires select options.")
                if kind == "select":
                    options = metadata.get("options")
                    if not isinstance(options, list) or not options:
                        raise ValueError(f"Select predictor '{name}' requires a nonempty options array.")
                    for option in options:
                        # Labels are presentation only; validate the original model value.
                        if isinstance(option, dict):
                            if not isinstance(option.get("label"), str) or not option["label"].strip():
                                raise ValueError(f"Options for '{name}' require nonempty labels.")
                            option = option.get("value")
                        valid = (isinstance(option, (int, float)) and not isinstance(option, bool)
                                 and math.isfinite(option)) if value_type == "number" else (
                                     isinstance(option, str) and bool(option.strip()))
                        if not valid:
                            raise ValueError(f"Options for '{name}' must contain {value_type} values.")
                features.append({**metadata, "name": name, "value_type": value_type})
            return jsonify({"features": features, "excluded_predictors": excluded, **target_range,
                            "prediction_interval": interval if interval_error is None else None,
                            "prediction_interval_error": interval_error})
        except (OSError, ValueError, TypeError, AttributeError) as error:
            return jsonify({"error": f"Cannot load form configuration: {error}"}), 500

    @app.errorhandler(500)
    def internal_error(error):
        # Flask logs the traceback; keep internal details out of the response.
        return jsonify({"error": "Unexpected server error while processing the request."}), 500

    @app.post("/predict")
    def predict():
        if interval_error is not None:
            return jsonify({"error": interval_error}), 503
        # A JSON object may omit any predictor; the fitted imputers fill gaps.
        if not request.get_data().strip():
            return jsonify({"error": "Request body is empty; provide a JSON object with predictor values."}), 400
        try:
            payload = request.get_json()
        except UnsupportedMediaType:
            return jsonify({"error": "Content-Type must be application/json."}), 400
        except BadRequest:
            return jsonify({"error": "Malformed JSON; check quotes, commas, and brackets."}), 400
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        values = []
        predictors_supplied = 0
        for name in predictors:
            value = payload.get(name)
            if value is None or (isinstance(value, str) and not value.strip()):
                values.append(np.nan)
                continue
            if name in numeric or name in numeric_categories:
                try:
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                except OverflowError:
                    valid = False
                expected = "a finite JSON number"
            else:
                valid = isinstance(value, str) and bool(value.strip())
                expected = "a nonempty string"
            if not valid:
                return jsonify({"error": f"Invalid '{name}': expected {expected}."}), 400
            if name in config.get("derived_binary_features", {}) and value >= 0 and value not in (0, 1):
                return jsonify({"error": f"Invalid '{name}': expected 0, 1, or null."}), 400
            values.append(value)
            predictors_supplied += 1

        # Preserve the exact training column names and order. Cleaning, imputation,
        # encoding, and any inverse target transform are already in model.pkl.
        row = pd.DataFrame([values], columns=predictors)
        # Preserve text-compatible columns even when every category is missing.
        for name in categorical:
            row[name] = row[name].astype(object)
        try:
            prediction = float(model.predict(row)[0])
            if not math.isfinite(prediction):
                return jsonify({"error": "Inputs produced a nonfinite prediction; check the submitted values."}), 400
        except (ValueError, OverflowError):
            return jsonify({"error": "Submitted predictor values could not be processed by the model."}), 400
        lower = max(0.0, prediction - interval["conformal_residual_quantile"])
        upper = prediction + interval["conformal_residual_quantile"]
        if not math.isfinite(upper) or not math.isfinite(lower) or upper < lower:
            return jsonify({"error": "Inputs produced an invalid prediction interval; check the submitted values."}), 400
        return jsonify({
            "predicted_value": prediction,
            "prediction_interval_95": {"lower": lower, "upper": upper},
            "predictors_supplied": predictors_supplied,
            "predictors_total": len(predictors),
            "predictors_ignored": [name for name in excluded if name in payload],
            **target_range,
        })

    return app


app = create_app()

if __name__ == "__main__":
    # Render uses Gunicorn; this also respects hosted PORT for direct execution.
    app.run(host="0.0.0.0" if os.environ.get("PORT") else "127.0.0.1",
            port=int(os.environ.get("PORT", "5000")), debug=False)
