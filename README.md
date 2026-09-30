# Housing Predictor

## Predictor discovery

Every column in the configured CSV is a predictor except `target` and `control_column` (default CONTROL). CSV order is preserved. No fixed predictor count or frontend list exists. Remove unwanted columns from the CSV.

`backend/model_settings.py` provides shared parsing and discovery for training, Flask, and tests. `model_config.json` selects data_file, target, control_column, and target_transform. Numeric columns default to numeric preprocessing; text columns default to categorical preprocessing. Optional feature_types overrides classify coded variables without controlling inclusion. Currently LOTSIZE and DIVISION are categorical. Absent override columns are ignored. Do not maintain numeric_features/categorical_features lists in JSON; these are derived at runtime.

Current predictors: UNITSIZE, BEDROOMS, BATHROOMS, YRBUILT, LOTSIZE, DIVISION, RATINGNH. This list describes the current data, not a required schema. MARKETVAL and CONTROL are excluded.

## Run

From the project folder, with Python 3.10 or newer:

```sh
python -m pip install -r backend/requirements.txt
python backend/train_model.py
python backend/app.py
```

In another terminal:

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:8000. After adding/removing CSV columns or changing types, retrain, restart Flask, and reload. No frontend changes are needed for numeric/categorical predictors.

## Training

Parsing supports ordinary CSV quoting and apostrophe-wrapped AHS codes. Missing/nonpositive housing targets are removed. Negative UNITSIZE/LOTSIZE codes become missing predictors; their rows are retained for imputation. Other survey special codes require research and explicit cleaning. UNITSIZE remains numeric, but its 1–9 values are not confirmed square footage. The CSV is never modified.

The pipeline retains HousingCodeCleaner, ColumnTransformer, numeric median imputation, categorical mode imputation/one-hot encoding, and LinearRegression. Preprocessing learns only from training rows. Unknown categories are accepted; all-missing columns use a zero fallback. Keep housing_preprocessing.py importable when loading the model.

Each run compares raw/log1p regression on one reproducible 80/20 split. Log predictions are inverted with expm1. Metrics and plots use dollars. Target summaries, percentiles, R², RMSE, MAE, median/max absolute errors and metric differences are printed. target_transform selects which complete pipeline is saved as backend/model.pkl. Raw/log plots are in backend/diagnostics; old unprefixed prediction plots are historical. Use matching dependency versions for training and serving.

## API and frontend

- GET /health: status and loaded-model fingerprint.
- GET /model-info: CSV-derived fields and generated frontend metadata.
- POST /predict: JSON containing every current predictor; returns numeric predicted_value in dollars.

CORS is enabled. Numeric inputs/categories require finite JSON numbers; text categories require nonempty strings. Nulls, booleans, arrays and objects are rejected. Invalid requests return JSON 400 errors; unexpected errors return 500. Startup detects saved-model/settings mismatches.

feature_schema.json contains optional display overrides, not an inclusion list. New columns automatically get labels, number controls or selects with observed categories. Override labels, help, bounds, steps or options as needed. Removed columns' metadata is ignored. Invalid overrides produce clear errors; all-missing categorical fields require explicit options. HTML contains only the field container; CSS handles styling; JavaScript renders model-info and manages validation, loading, errors and currency formatting.

## Verification

```sh
python backend/test_predict.py --audit-only
python backend/test_predict.py
```

Use --base-url http://127.0.0.1:5001 for another port. The test constructs all CSV-derived predictors automatically, prints the exact payload/status/full JSON/prediction/errors, and asserts model columns/types, dataset/configuration fingerprints, API model fingerprint, prediction parity and missing-field rejection. Audit-only needs no running API. Restart stale Flask processes after retraining.
