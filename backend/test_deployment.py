"""Deployment checks that do not create services, retrain, or change the model."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parent


class DeploymentTests(unittest.TestCase):
    def test_case_sensitive_dataset_and_runtime_files(self):
        settings = json.loads((BACKEND / "model_config.json").read_text())
        self.assertIn(settings["data_file"], [path.name for path in (BACKEND.parent / "data").iterdir()])
        for name in ("model.pkl", "model_config.json", "feature_schema.json", "housing_preprocessing.py"):
            self.assertTrue((BACKEND / name).is_file(), name)

    def test_pages_cors_and_prediction(self):
        import app as api
        origin = "https://manofsteelix2.github.io"
        with patch.dict(os.environ, {"CORS_ORIGINS": origin}):
            app = api.create_app()
        self.assertFalse(app.debug)
        client = app.test_client()
        info = client.get("/model-info", headers={"Origin": origin})
        self.assertEqual(info.status_code, 200, info.json)
        self.assertEqual(info.headers.get("Access-Control-Allow-Origin"), origin)
        preflight = client.options("/predict", headers={"Origin": origin,
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(preflight.headers.get("Access-Control-Allow-Origin"), origin)
        self.assertIn("content-type", preflight.headers.get("Access-Control-Allow-Headers", "").lower())
        self.assertIsNone(client.get("/health", headers={"Origin": "https://unlisted.example"})
                          .headers.get("Access-Control-Allow-Origin"))
        response = client.post("/predict", json={}, headers={"Origin": origin})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), origin)
        self.assertIn("prediction_interval_95", response.json)

    def test_import_with_unrelated_working_directory(self):
        code = (f"import sys, os; sys.path.insert(0, {str(BACKEND)!r}); "
                "import app; c=app.app.test_client(); "
                "assert c.get('/health').status_code == 200; "
                "assert c.get('/model-info').status_code == 200; "
                "assert c.post('/predict',json={}).status_code == 200")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tempfile.gettempdir(),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
