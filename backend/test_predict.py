
"""Test optional inputs and invalid values: python backend/test_predict.py."""

import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from model_settings import load_config, read_dataset


BACKEND_DIR = Path(__file__).resolve().parent


def send_prediction(base_url, payload):
    request = Request(
        base_url.rstrip("/") + "/predict",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        response = error  # Report API errors before checking the expected status.
    with response:
        status = response.code
        body = json.loads(response.read().decode("utf-8"))
    print(f"Status: {status}")
    print("JSON:", json.dumps(body, indent=2))
    return status, body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    args = parser.parse_args()

    # Use the artifact's final subset; no complete predictor row is required.
    config = load_config(BACKEND_DIR / "model_config.json")
    model = joblib.load(BACKEND_DIR / "model.pkl")
    predictors = model.training_metadata_.get("final_predictors", config["predictors"])
    data = read_dataset(BACKEND_DIR.parent / "data" / config["data_file"])
    values = {}
    for name in predictors:
        observed = data[name].dropna()
        values[name] = observed.iloc[0] if not observed.empty else None
    payload = json.loads(pd.Series(values).to_json())
    # Omit predictors of both types when available, retaining one for a partial request.
    candidates = list(dict.fromkeys(
        [name for name in config["numeric_features"] if name in predictors][:2]
        + [name for name in config["categorical_features"] if name in predictors][:2] + predictors
    ))
    omitted = set(candidates[:min(4, max(0, len(predictors) - 1))])
    incomplete = {key: value for key, value in payload.items() if key not in omitted}
    for label, values in (("All predictors", payload),
                          ("Several omitted predictors", incomplete),
                          ("One predictor", {predictors[0]: payload[predictors[0]]}),
                          ("No predictors", {}),
                          ("Null and blank predictors", dict.fromkeys(predictors, None) | {
                              predictors[0]: "", predictors[-1]: "   "
                          })):
        print(f"\n{label} request:", json.dumps(values))
        status, body = send_prediction(args.base_url, values)
        prediction = body.get("predicted_value")
        if status != 200 or isinstance(prediction, bool) or not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
            raise AssertionError(f"{label}: expected 200 and a finite numeric predicted_value.")
        supplied = sum(value is not None and not (isinstance(value, str) and not value.strip())
                       for value in values.values())
        if body.get("predictors_supplied") != supplied or body.get("predictors_total") != len(predictors):
            raise AssertionError(f"{label}: incorrect predictor counts.")
        interval = body.get("prediction_interval_95", {})
        q = model.training_metadata_["prediction_interval"]["conformal_residual_quantile"]
        if (interval.get("lower") != max(0, prediction - q)
                or interval.get("upper") != prediction + q):
            raise AssertionError(f"{label}: interval must use the saved conformal residual quantile.")
    numeric = [name for name in config["numeric_features"] if name in predictors]
    if numeric:
        print("\nInvalid provided numeric value:")
        status, body = send_prediction(args.base_url, {numeric[0]: "not a number"})
        if status != 400 or numeric[0] not in body.get("error", ""):
            raise AssertionError("Expected 400 and an error naming the invalid numeric predictor.")
    excluded = [name for name in config["predictors"] if name not in predictors]
    if excluded:
        status, body = send_prediction(args.base_url, {name: "ignored" for name in excluded})
        if status != 200 or body.get("predictors_ignored") != excluded:
            raise AssertionError("Excluded predictors should be ignored and reported.")
    print("\nAll prediction API smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except URLError as error:
        raise SystemExit(f"Cannot reach the API. Start backend/app.py first. Details: {error}")
