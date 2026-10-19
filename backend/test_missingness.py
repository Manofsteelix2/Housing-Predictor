"""Run with python -B -m unittest discover -s backend -p test_missingness.py."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model_settings import load_config
from housing_preprocessing import HousingCodeCleaner
from train_model import build_pipeline, calibrate_prediction_interval, load_data, select_predictors, train_model


class MissingnessTests(unittest.TestCase):
    def test_finite_sample_conformal_quantile(self):
        calibration = calibrate_prediction_interval(np.arange(1, 41), np.zeros(40))
        self.assertEqual(calibration["quantile_rank"], 39)  # ceil(41 * .95)
        self.assertEqual(calibration["conformal_residual_quantile"], 39)
        self.assertEqual(calibration["calibration_sample_size"], 40)
        self.assertEqual(calibration["coverage_level"], .95)
        self.assertEqual(calibration["method"], "split_conformal_absolute_residual")
        self.assertEqual(calibrate_prediction_interval(np.arange(19), np.zeros(19))
                         ["conformal_residual_quantile"], 18)
        with self.assertRaisesRegex(ValueError, "19 calibration"):
            calibrate_prediction_interval(np.arange(18), np.zeros(18))
        with self.assertRaisesRegex(ValueError, "finite"):
            calibrate_prediction_interval([np.nan] * 20, np.zeros(20))

    def test_target_range_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.csv"
            pd.DataFrame({"MARKETVAL": [10000], "A": [1]}).to_csv(dataset, index=False)
            config_path = root / "model_config.json"
            settings = {"target": "MARKETVAL", "data_file": "data.csv"}
            for minimum, maximum in ((20000, 10000), (-1, 10000), (True, 10000),
                                     (10000, float("inf")), ("10000", 5000000)):
                config_path.write_text(json.dumps({**settings, "target_min": minimum,
                                                    "target_max": maximum}), encoding="utf8")
                with self.subTest(minimum=minimum, maximum=maximum), self.assertRaises(ValueError):
                    load_config(config_path, dataset)
            config_path.write_text(json.dumps({**settings, "target_min": 10000,
                                                "target_max": 10000}), encoding="utf8")
            config = load_config(config_path, dataset)
            self.assertEqual(config["target_min"], config["target_max"])

    def test_negative_codes_across_all_predictor_types(self):
        raw = pd.DataFrame({"AREA": [-1, 0, 3, np.nan],
                            "CODE": [-9, 1, 2, 0],
                            "TEXT": ["-6", "north", "0", "south"]})
        cleaned = HousingCodeCleaner().transform(raw)
        self.assertEqual(cleaned.shape, raw.shape)
        self.assertTrue(cleaned.iloc[0].isna().all())
        self.assertEqual(cleaned.loc[1, "AREA"], 0)
        self.assertEqual(cleaned.loc[1, "TEXT"], "north")
        self.assertEqual(cleaned.loc[2, "TEXT"], "0")
        self.assertEqual(raw.loc[0, "AREA"], -1)  # Cleaner does not mutate inputs.
        config = {"numeric_features": ["AREA"], "categorical_features": ["CODE"]}
        model = build_pipeline(config).fit(raw[["AREA", "CODE"]], [100, 200, 300, 400])
        negative = pd.DataFrame({"AREA": [-99], "CODE": [-9]})
        missing = pd.DataFrame({"AREA": [np.nan], "CODE": [np.nan]})
        np.testing.assert_allclose(model.predict(negative), model.predict(missing))
        self.assertEqual(model.named_steps["preprocessor"].named_transformers_["numeric"]
                         .named_steps["imputer"].statistics_.tolist(), [1.5])

    def test_threshold_uses_training_rows_and_preserves_order(self):
        config = {"predictors": ["B", "A", "C"], "numeric_features": ["B", "A", "C"],
                  "categorical_features": [], "max_missing_fraction": 0.5}
        training = pd.DataFrame({"B": [1, np.nan], "A": [np.nan, np.nan], "C": [2, 3]})
        full = pd.concat([training, pd.DataFrame({"B": [np.nan] * 8, "A": [1] * 8, "C": [1] * 8})])
        with contextlib.redirect_stdout(io.StringIO()):
            fitted, report, excluded = select_predictors(full, training, config)
        self.assertEqual(fitted["predictors"], ["B", "C"])
        self.assertEqual(excluded, ["A"])
        self.assertEqual(report["B"]["missing"], 9)
        self.assertEqual(report["B"]["nonmissing"], 1)
        self.assertEqual(report["B"]["percent_missing"], 90)
        self.assertFalse(report["B"]["excluded"])  # Exactly 50% in training is retained.
        with self.assertRaisesRegex(ValueError, "nonempty"):
            select_predictors(training, training, {**config, "predictors": ["A"],
                                                  "numeric_features": ["A"], "max_missing_fraction": 0})

    def test_imputers_learn_training_median_and_missing_category(self):
        config = {"numeric_features": ["AREA"], "categorical_features": ["CODE"],
                  "add_missing_indicators": True}
        training = pd.DataFrame({"AREA": [1., 3., np.nan, 9.], "CODE": [1., 2., np.nan, 1.]})
        model = build_pipeline(config).fit(training, [100, 300, 200, 900])
        preprocessor = model.named_steps["preprocessor"]
        numeric = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        self.assertEqual(numeric.statistics_.tolist(), [3.])
        self.assertEqual(numeric.indicator_.features_.tolist(), [0])
        categorical = preprocessor.named_transformers_["categorical"]
        self.assertEqual(categorical.named_steps["imputer"].strategy, "constant")
        self.assertIn("__MISSING__", categorical.named_steps["encoder"].categories_[0])
        missing = pd.DataFrame({"AREA": [np.nan], "CODE": [np.nan]})
        observed = pd.DataFrame({"AREA": [3.], "CODE": [1]})
        self.assertTrue(np.isfinite(model.predict(missing)).all())
        self.assertFalse(np.array_equal(preprocessor.transform(missing), preprocessor.transform(observed)))
        np.testing.assert_allclose(model.predict(pd.DataFrame({"AREA": [3.], "CODE": [1.]})),
                                   model.predict(observed))

    def test_training_roundtrip_and_api_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "backend"
            backend.mkdir()
            (root / "data").mkdir()
            data_path = root / "data" / "sample.csv"
            config_path = backend / "model_config.json"
            settings = {"target": "MARKETVAL", "data_file": "sample.csv", "target_transform": "log1p",
                        "target_min": 1000, "target_max": 3975,
                        "feature_types": {"CODE": "categorical"}, "max_missing_fraction": 0.75,
                        "add_missing_indicators": True}
            config_path.write_text(json.dumps(settings), encoding="utf8")
            data = pd.DataFrame({
                "CONTROL": range(120), "MARKETVAL": [1000 + i * 25 for i in range(120)],
                "A": [i if i % 2 else np.nan for i in range(120)],
                "B": [np.nan if i % 2 else i for i in range(120)],
                "CODE": [np.nan if i % 3 == 0 else i % 2 for i in range(120)],
                "SPARSE": [np.nan] * 119 + [1.],
            })
            # Invalid targets are never imputed; every valid target survives, even
            # though no row has a complete set of predictors.
            data.loc[120] = [120, np.nan, 1, 1, 1, 1]
            data.loc[121] = [121, 0, 1, 1, 1, 1]
            data.loc[122] = [122, np.inf, 1, 1, 1, 1]
            data.loc[123] = [123, -5, 1, 1, 1, 1]
            data.loc[124] = [124, "invalid", 1, 1, 1, 1]
            data.loc[125] = [125, 999, 1, 1, 1, 1]
            data.loc[126] = [126, 3976, 1, 1, 1, 1]
            data.loc[1, "A"] = -99
            data.loc[2, "CODE"] = -6
            data.to_csv(data_path, index=False)
            config = load_config(config_path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                features, target = load_data(data_path, config)
                model, metrics = train_model(data_path, config_path, backend / "model.pkl", backend / "plots")
            self.assertEqual(len(target), 120)
            self.assertEqual(target.min(), 1000)  # Both range endpoints survive.
            self.assertEqual(target.max(), 3975)
            self.assertIn("Excluded 1 observations below $1,000.", output.getvalue())
            self.assertIn("Excluded 1 observations above $3,975.", output.getvalue())
            self.assertIn("not missing-data cleaning", output.getvalue())
            self.assertEqual(len(features), 120)
            self.assertTrue(pd.isna(features.loc[1, "A"]))
            self.assertTrue(pd.isna(features.loc[2, "CODE"]))
            self.assertTrue(features.isna().any(axis=1).all())
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            saved = joblib.load(backend / "model.pkl")
            self.assertEqual(saved.training_metadata_["target_min"], 1000)
            self.assertEqual(saved.training_metadata_["target_max"], 3975)
            self.assertEqual(saved.training_metadata_["final_predictors"], ["A", "B", "CODE"])
            self.assertEqual(saved.training_metadata_["excluded_predictors"], ["SPARSE"])
            remaining, test = train_test_split(features, test_size=0.2, random_state=42)
            training, calibration = train_test_split(remaining, test_size=0.25, random_state=43)
            self.assertFalse(set(training.index) & set(calibration.index))
            self.assertFalse(set(training.index) & set(test.index))
            self.assertFalse(set(calibration.index) & set(test.index))
            self.assertEqual(len(training) + len(calibration) + len(test), len(features))
            imputer = saved.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
            np.testing.assert_allclose(imputer.statistics_, training[["A", "B"]].median())
            final = saved.training_metadata_["final_predictors"]
            expected = calibrate_prediction_interval(target.loc[calibration.index], saved.predict(calibration[final]))
            self.assertEqual(saved.training_metadata_["prediction_interval"], expected)
            test_predictions = saved.predict(test[final])
            q = expected["conformal_residual_quantile"]
            lower, upper = np.maximum(0, test_predictions - q), test_predictions + q
            evaluation = saved.training_metadata_["prediction_interval_test_evaluation"]
            self.assertAlmostEqual(evaluation["observed_coverage"],
                                   np.mean((target.loc[test.index] >= lower) & (target.loc[test.index] <= upper)))
            self.assertAlmostEqual(evaluation["average_interval_width"], np.maximum(0, upper - lower).mean())
            self.assertIn("rows retained", output.getvalue())
            import app as api
            with patch.object(api, "BACKEND_DIR", backend):
                client = api.create_app().test_client()
                info = client.get("/model-info")
                self.assertEqual(info.status_code, 200, info.json)
                self.assertEqual(info.json["target_min"], 1000)
                self.assertEqual(info.json["target_max"], 3975)
                self.assertEqual(info.json["prediction_interval"], expected)
                self.assertEqual([field["name"] for field in info.json["features"]], ["A", "B", "CODE"])
                self.assertTrue(all(field["required"] is False for field in info.json["features"]))
                for payload in ({}, {"A": 3}, {"CODE": 1}, {"CODE": ""}, {"SPARSE": {"invalid": True}}):
                    response = client.post("/predict", json=payload)
                    self.assertEqual(response.status_code, 200, response.json)
                    self.assertTrue(np.isfinite(response.json["predicted_value"]))
                    self.assertEqual(response.json["predictors_total"], 3)
                    bounds = response.json["prediction_interval_95"]
                    self.assertAlmostEqual(bounds["lower"], max(0, response.json["predicted_value"] - q))
                    self.assertAlmostEqual(bounds["upper"], response.json["predicted_value"] + q)
                self.assertEqual(response.json["predictors_ignored"], ["SPARSE"])
                self.assertEqual(response.json["target_min"], 1000)
                self.assertEqual(response.json["target_max"], 3975)
                negative = client.post("/predict", json={"A": -99, "CODE": -6})
                missing = client.post("/predict", json={})
                self.assertEqual(negative.status_code, 200, negative.json)
                self.assertAlmostEqual(negative.json["predicted_value"], missing.json["predicted_value"])
                self.assertEqual(client.post("/predict", json={"CODE": "invalid"}).status_code, 400)
                self.assertEqual(client.get("/health").status_code, 200)
                # Legacy metadata must not silently produce an uncalibrated estimate.
                legacy = copy.deepcopy(saved)
                del legacy.training_metadata_["prediction_interval"]
                with patch.object(api.joblib, "load", return_value=legacy):
                    legacy_client = api.create_app().test_client()
                legacy_response = legacy_client.post("/predict", json={})
                self.assertEqual(legacy_response.status_code, 503)
                self.assertIn("Retrain", legacy_response.json["error"])
                self.assertEqual(legacy_client.get("/health").status_code, 200)
                # Lower bound clipping is applied using the stored calibration only.
                wide = copy.deepcopy(saved)
                wide.training_metadata_["prediction_interval"]["conformal_residual_quantile"] = 1e9
                with patch.object(api.joblib, "load", return_value=wide):
                    wide_client = api.create_app().test_client()
                self.assertEqual(wide_client.post("/predict", json={}).json["prediction_interval_95"]["lower"], 0)


if __name__ == "__main__":
    unittest.main()
