"""Verify source-code derivation, binary API inputs, and artifact consistency."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from housing_preprocessing import HousingCodeCleaner
from model_settings import derive_binary_features, load_config, read_dataset
from train_model import build_pipeline


BACKEND = Path(__file__).resolve().parent
DEFINITIONS = json.loads((BACKEND / "model_config.json").read_text())["derived_binary_features"]


class BinaryAmenityTests(unittest.TestCase):
    def test_source_rules_missing_and_idempotence(self):
        codes = [0, 1, 2, 3, 4, 5, 6, -6, -9, None, ".M", ".N", "", float("inf")]
        raw = pd.DataFrame({definition["source"]: codes for definition in DEFINITIONS.values()})
        binary = HousingCodeCleaner(derived_binary_features=DEFINITIONS).transform(raw)
        self.assertEqual(list(binary), ["WASHER_PRESENT", "PORCH_PRESENT", "FIREPLACE_PRESENT", "DINING_PRESENT"])
        for name, definition in DEFINITIONS.items():
            expected = [float(code in definition["yes_values"]) for code in range(7)] + [np.nan] * 7
            np.testing.assert_allclose(binary[name], expected, equal_nan=True)
        pd.testing.assert_frame_equal(binary, derive_binary_features(binary, DEFINITIONS))
        with self.assertRaises(ValueError):
            derive_binary_features(raw.assign(WASHER_PRESENT=1), DEFINITIONS)

    def test_discovery_replaces_sources_in_csv_order(self):
        config = load_config(BACKEND / "model_config.json")
        raw = read_dataset(BACKEND.parent / "data" / config["data_file"])
        binary = read_dataset(BACKEND.parent / "data" / config["data_file"], DEFINITIONS)
        replacements = {rule["source"]: name for name, rule in DEFINITIONS.items()}
        expected = [replacements.get(name, name) for name in raw if name not in (config["target"], config["control_column"])]
        self.assertEqual(config["predictors"], expected)
        self.assertFalse(set(replacements) & set(binary))
        for name in DEFINITIONS:
            self.assertTrue(set(binary[name].dropna().unique()) <= {0, 1})
            self.assertIn(name, config["numeric_features"])

    def test_pipeline_raw_binary_and_saved_predictions_match(self):
        raw = pd.DataFrame({rule["source"]: [0, 1, 2, 3, 4, 5, -9, None] * 4
                            for rule in DEFINITIONS.values()})
        binary = derive_binary_features(raw, DEFINITIONS)
        config = {"numeric_features": list(DEFINITIONS), "categorical_features": [],
                  "derived_binary_features": DEFINITIONS}
        pipeline = build_pipeline(config).fit(binary, np.arange(len(binary)) * 1000 + 100000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            joblib.dump(pipeline, path)
            saved = joblib.load(path)
            np.testing.assert_allclose(saved.predict(raw), saved.predict(binary))
            cleaned = saved.named_steps["code_cleaner"].transform(binary)
            pd.testing.assert_frame_equal(cleaned, binary)

    def test_retrained_api_binary_inputs(self):
        from app import app
        client = app.test_client()
        info = client.get("/model-info")
        self.assertEqual(info.status_code, 200, info.json)
        fields = {field["name"]: field for field in info.json["features"]}
        schema = json.loads((BACKEND / "feature_schema.json").read_text())
        for name, rule in DEFINITIONS.items():
            self.assertNotIn(rule["source"], fields)
            self.assertEqual(fields[name]["options"], schema[name]["options"])
            self.assertFalse(fields[name]["required"])
            for value in (0, 1, None, "", -9):
                response = client.post("/predict", json={name: value})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(np.isfinite(response.json["predicted_value"]))
            missing = client.post("/predict", json={name: None}).json["predicted_value"]
            for value in ("", -9):
                self.assertAlmostEqual(client.post("/predict", json={name: value}).json["predicted_value"], missing)
            for value in (2, 3, True):
                response = client.post("/predict", json={name: value})
                self.assertEqual(response.status_code, 400, response.json)


if __name__ == "__main__":
    unittest.main()
