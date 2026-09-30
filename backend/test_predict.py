
"""Send two smoke-test requests to a running API: python backend/test_predict.py."""

import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


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
        response = error  # A 400 response is expected for the missing-field test.
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

    # Use configured columns and a complete dataset row, so new predictors do
    # not require rewriting this test or inventing category values.
    with (BACKEND_DIR / "model_config.json").open(encoding="utf-8-sig") as file:
        config = json.load(file)
    predictors = config["numeric_features"] + config["categorical_features"]
    data = pd.read_csv(BACKEND_DIR.parent / "data" / "housingB.csv")
    complete_rows = data[predictors].dropna()
    if complete_rows.empty:
        raise ValueError("Test request needs at least one complete predictor row in housingB.csv.")
    payload = json.loads(complete_rows.iloc[0].to_json())
    print("Valid request:", json.dumps(payload))
    status, body = send_prediction(args.base_url, payload)
    prediction = body.get("predicted_value")
    if status != 200 or isinstance(prediction, bool) or not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
        raise AssertionError("Expected 200 and a finite numeric predicted_value.")

    missing_field = predictors[0]
    incomplete = {key: value for key, value in payload.items() if key != missing_field}
    print(f"\nMissing-field request (omitting {missing_field}):")
    status, body = send_prediction(args.base_url, incomplete)
    if status != 400 or missing_field not in body.get("error", ""):
        raise AssertionError("Expected 400 and an error naming the missing predictor.")
    print("\nBoth prediction API smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except URLError as error:
        raise SystemExit(f"Cannot reach the API. Start backend/app.py first. Details: {error}")
