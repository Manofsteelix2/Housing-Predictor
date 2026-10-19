# Housing Predictor

## Predictor discovery

Every column in the configured CSV is a candidate predictor except `target` and `control_column` (default CONTROL). CSV order is preserved. Training may exclude columns with excessive missingness; it never removes rows because predictors are missing. No fixed predictor count or frontend list exists.

`backend/model_settings.py` provides shared parsing and discovery for training, Flask, and tests. `model_config.json` selects data_file, target, control_column, and target_transform. Numeric columns default to numeric preprocessing; text columns default to categorical preprocessing. Optional feature_types overrides classify coded variables without controlling inclusion. Currently LOTSIZE and DIVISION are categorical. Absent override columns are ignored. Do not maintain numeric_features/categorical_features lists in JSON; these are derived at runtime.

Every source predictor remains eligible in CSV order. Configured `derived_binary_features` replace their source columns in that same position; original and derived versions are never included together. MARKETVAL and CONTROL are excluded.

The configured amenity features are WASHER_PRESENT (source WASHER = 1), PORCH_PRESENT (PORCH = 1), FIREPLACE_PRESENT (FIREPLACE in 1–3), and DINING_PRESENT (DINING in 1–5). These source codes map to 1; other finite nonnegative source values map to 0. Negative, blank, nonnumeric survey missing codes and nonfinite values map to NaN. Shared dataset parsing applies the definitions before discovery and training missingness selection, and the saved HousingCodeCleaner retains the same definitions. It can derive raw source columns for direct pipeline use or accept already-derived binary columns without recoding them. The CSV remains unchanged.

The frontend uses only the derived names for these amenities, with Yes = 1, No = 0, and Not specified = null. The API rejects nonnegative derived values other than 0 or 1; negative codes remain missing. Change derivation definitions only with retraining and a Flask restart.

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

Parsing supports ordinary CSV quoting and apostrophe-wrapped AHS codes. Missing, nonnumeric, nonfinite, and nonpositive housing targets are removed and never imputed. Every negative numeric predictor value becomes NaN, including negative codes in categorical columns; rows are retained for pipeline imputation. This shared rule applies during training and prediction, before missingness reporting and feature selection. UNITSIZE remains numeric, but its 1–9 values are not confirmed square footage. The CSV is never modified.

The pipeline retains HousingCodeCleaner, ColumnTransformer, and LinearRegression. Numeric missing values use training-set medians. `add_missing_indicators` (default true) adds indicators for numeric columns missing during fitting. Categorical values are normalized to strings, missing values become the distinct `__MISSING__` category, and one-hot encoding handles unknown values. Keep housing_preprocessing.py importable when loading the model.

MARKETVAL is never imputed: missing, nonpositive, and nonfinite targets are excluded. Predictor NaNs remain until pipeline preprocessing. A missingness report lists missing counts, percentages, and nonmissing counts for every candidate among valid-target rows. `max_missing_fraction` (default 0.50, range 0–1) excludes predictors whose **training-partition** missing fraction strictly exceeds the threshold. Test rows do not influence exclusions or medians. No predictor-based row deletion is used. If every predictor is excluded, training stops with a clear error. At threshold 1, all-missing columns can be retained; numeric columns use sklearn's zero fallback because no median exists.

The final predictor list, excluded predictors, missingness report, and numeric categorical input types are saved in `model.pkl` training metadata alongside the full resolved configuration. Retrain and restart Flask after changing missingness settings. The API uses the saved final subset and verifies its column order against the fitted preprocessor.

`target_min` and `target_max` in `model_config.json` intentionally restrict the model's training scope (currently $10,000–$3,000,000, inclusive). After invalid/missing-target cleaning and before the train/test split, training prints the configured range, separate counts below/above it, and the remaining row count. These exclusions are model-scope restrictions, not missing-data cleaning. Both bounds are saved directly in training metadata and exposed by `/model-info` and `/predict`; the frontend displays the loaded model's training range. Predictions are not clipped to the range. Bounds must be finite nonnegative numbers with minimum <= maximum; omitted/null bounds leave that side unrestricted. Change the settings, retrain, and restart Flask to test a different range.

Each run first reserves 20% as an independent test set (`random_state=42`), then splits the remaining rows into fitting/calibration partitions (`test_size=0.25`, `random_state=43`), giving approximately 60% fitting, 20% calibration, and 20% testing. Feature exclusions, imputers, encoders, and regressors learn only from fitting rows. Raw/log1p models share all three partitions; target_transform selects the saved model without using test results to select it. Log predictions are inverted with expm1. Metrics and plots use dollars. Target summaries, percentiles, R², RMSE, MAE, median/max absolute errors and metric differences are preserved. Raw/log plots are in backend/diagnostics; old unprefixed prediction plots are historical. Use matching dependency versions for training and serving.

### 95% prediction intervals

Each fitted pipeline predicts the calibration partition on the original dollar scale. Split conformal uses absolute residuals and the `ceil((n_calibration + 1) * 0.95)`-th smallest residual (a finite-sample higher order statistic, without interpolation). At least 19 calibration rows are required for a finite bound. The saved pipeline is not refitted after calibration. Test rows are used only to report coverage and average displayed interval width; they never set the residual threshold. See the [split conformal method reference](https://arxiv.org/abs/2107.07511).

`training_metadata_["prediction_interval"]` stores coverage_level, calibration_sample_size, conformal_residual_quantile, quantile_rank, and method (`split_conformal_absolute_residual`). Metadata also records split sizes/seeds and independent test interval results. The API constructs `prediction_interval_95` as `max(0, predicted_value - q)` to `predicted_value + q`, using the saved dollar residual threshold; no calibration occurs during requests. The upper bound is not clipped to target_max. A legacy or invalid calibration yields a clear HTTP 503 retraining error from `/predict`, while `/health` and `/model-info` remain available for a settings-compatible artifact.

The frontend gives the point estimate greater visual prominence than the prediction range and explains the 2023 AHS source and absence of precise local market conditions. Coverage is marginal for comparable observations under exchangeability within the modeled target scope; it is not a guarantee for each location, home, subgroup, or arbitrary pattern of omitted inputs. The absolute-residual method uses the same dollar half-width for every request, with displayed width reduced when the lower bound is clipped at zero.

## API and frontend

The optional DIVISION select uses the nine Census division names from feature_schema.json. Its clickable SVG map selects the same control, highlights every state in that division, and includes Alaska, Hawaii and DC. Dropdown changes and blank selections update the map. The selected numeric code (1–9), or null when unspecified, is sent unchanged to the API; location is modeled at Census division level rather than exact local market location. State membership lives alongside division options in the schema. Local geometry in frontend/census-map.js is derived from [us-atlas](https://github.com/topojson/us-atlas), using Census boundaries; its license is in frontend/us-map-LICENSE.txt. The map requires no runtime network service or mapping library.

- GET /health: status and loaded-model fingerprint.
- GET /model-info: fields used by the fitted model, optional display metadata, excluded_predictors, training target bounds, and prediction_interval calibration metadata (or prediction_interval_error).
- POST /predict: JSON with any subset of predictors, including `{}`; returns predicted_value in dollars, prediction_interval_95 (lower/upper), predictors_supplied, predictors_total (fitted predictor count), and predictors_ignored (excluded fields present in the request).

CORS is enabled. Missing keys, nulls, empty strings, and whitespace-only strings become NaN for fitted pipeline imputation. Supplied numeric inputs/categories require finite JSON numbers; supplied text categories require strings. Booleans, arrays and objects are rejected for active predictors. Fields excluded during fitting are ignored even if their supplied values are invalid. Invalid requests return JSON 400 errors; unexpected errors return 500. Startup detects saved-model/settings mismatches.

feature_schema.json contains optional display overrides, not an inclusion list. Active columns automatically get labels, number controls or selects with observed categories. Fields default to `required: false`; an explicit `required: true` overrides frontend validation only. Blank optional fields submit null before numeric conversion. Select options may be scalar codes or objects with separate `value` and `label`; only values are submitted. Negative and survey missing codes are excluded from displayed choices. Override labels, help, bounds, steps or options as needed. Removed/excluded columns' metadata is ignored. Invalid overrides produce clear errors; all-missing categorical fields require explicit options if retained. HTML contains only the field container; CSS handles styling; JavaScript renders model-info and manages validation, loading, errors and currency formatting.

## Verification

```sh
python -B -m unittest discover -s backend -p test_missingness.py
python -B -m unittest discover -s backend -p test_binary_amenities.py
python backend/test_predict.py
node frontend/test_optional_inputs.cjs
```

Use --base-url http://127.0.0.1:5001 for another port. The API smoke test covers full, partial, single-field, empty, null/blank, invalid numeric, and excluded-field requests. It uses observed per-column values without requiring any complete predictor row. The unittest suite checks row retention, training-only selection/medians, missing categories/indicators, artifact roundtrips, and API compatibility using synthetic data. Restart stale Flask processes after retraining.
