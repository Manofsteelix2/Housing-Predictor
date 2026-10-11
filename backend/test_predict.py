
"""Test optional inputs and invalid values: python backend/test_predict.py."""

import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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

    # Use configured columns and a complete dataset row, so new predictors do
    # not require rewriting this test or inventing category values.
    config = load_config(BACKEND_DIR / "model_config.json")
    predictors = config["predictors"]
    data = read_dataset(BACKEND_DIR.parent / "data" / config["data_file"])
    complete_rows = data[predictors].dropna()
    if complete_rows.empty:
        raise ValueError(f"Test request needs at least one complete predictor row in {config['data_file']}.")
    payload = json.loads(complete_rows.iloc[0].to_json())
    # Omit predictors of both types when available, retaining one for a partial request.
    candidates = list(dict.fromkeys(
        config["numeric_features"][:2] + config["categorical_features"][:2] + predictors
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
    numeric = config["numeric_features"]
    if numeric:
        print("\nInvalid provided numeric value:")
        status, body = send_prediction(args.base_url, {numeric[0]: "not a number"})
        if status != 400 or numeric[0] not in body.get("error", ""):
            raise AssertionError("Expected 400 and an error naming the invalid numeric predictor.")
    print("\nAll prediction API smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except URLError as error:
        raise SystemExit(f"Cannot reach the API. Start backend/app.py first. Details: {error}")
