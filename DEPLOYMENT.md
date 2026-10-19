# Deployment preparation (nothing deployed)

## Render backend

Create a Python web service from this repository when ready. Keep **Root Directory empty (repository root)**. Render excludes files outside a configured root directory, and the API needs `data/housingG.csv` alongside `backend/`.

Build command:

```sh
pip install -r backend/requirements.txt
```

Start command:

```sh
gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

`backend/app.py` exports the Flask instance `app`; Gunicorn imports that instance without running Flask's development server. The bind address and Render-provided PORT make it reachable by Render's proxy. Health check: `/health`. `render.yaml` contains the same settings with automatic deploys disabled. Importing a Blueprint later still creates an initial deployment; no Blueprint has been imported here.

Use Python **3.14.8**, matching `.python-version` and the fitted pipeline environment. Requirements pin the installed model dependencies to avoid silently loading an sklearn pickle with a different version. Gunicorn is installed only on non-Windows platforms. The Gunicorn process itself must be tested on Linux; Windows checks use Flask's test client.

Set `CORS_ORIGINS=https://manofsteelix2.github.io`. This is the origin for `https://manofsteelix2.github.io/Housing-Predictor/`; do not include the repository path or trailing slash in CORS_ORIGINS. Change it if the Pages account or custom domain changes. Multiple origins can be comma-separated. Local Flask defaults also permit the two localhost frontend origins for development. Credentials are not enabled.

## Required runtime files

Commit and preserve these paths, including exact capitalization:

- `backend/app.py`
- `backend/model.pkl` (existing fitted artifact; never regenerate during deployment)
- `backend/model_config.json`
- `backend/feature_schema.json`
- `backend/model_settings.py`
- `backend/housing_preprocessing.py` (required to unpickle the custom transformer)
- `backend/requirements.txt`
- `data/housingG.csv`

The tracked CSV was renamed from `HousingG.csv` to `housingG.csv` to match the existing config on Linux. The data bytes and model config are unchanged. Model/schema/dataset paths are resolved from `__file__`, not the shell's current directory. Do not set Render's root to `backend` or omit the dataset: startup and `/model-info` both read it.

Also commit `.gitignore`, `.python-version`, `render.yaml`, `DEPLOYMENT.md`, `backend/test_deployment.py`, and the complete `frontend/` assets. `.gitignore` does not exclude the model, JSON configs, CSVs, or frontend. Previously tracked bytecode remains tracked until explicitly removed; it is not needed for deployment.

## GitHub Pages frontend

The only backend URL setting is `API_BASE` near the top of `frontend/script.js`. It currently contains the requested placeholder:

```js
const API_BASE = "https://YOUR-RENDER-SERVICE.onrender.com";
```

After Render provides the live URL, replace that value with the service's HTTPS origin, without a trailing slash. Both `/model-info` and `/predict` derive from it; production requests never use localhost. The placeholder intentionally cannot serve predictions until replaced. For local development, temporarily change this same constant to your local API origin and restore the HTTPS value before publication.

Commit `.github/workflows/pages.yml`. It publishes only `frontend/` (not model/data files) and has **only a manual `workflow_dispatch` trigger**. It refuses to publish while API_BASE is the placeholder, localhost, or non-HTTPS. When ready, select **Settings → Pages → Source: GitHub Actions**, then manually run the workflow. Nothing runs or deploys just because these files were prepared.

The resulting project site should be `https://manofsteelix2.github.io/Housing-Predictor/`. All frontend asset references are relative, so the repository path works without rewriting links.

## Checks before the eventual deployment

```sh
python -B -m unittest discover -s backend -p test_deployment.py
node frontend/test_optional_inputs.cjs
```

The frontend test uses the actual Flask test client through a local Python process; set the PYTHON environment variable if the `python` executable is not on PATH. It does not contact or deploy the Render placeholder. After deployment, check `/health`, `/model-info`, a browser prediction, the response's prediction interval, and the Pages-origin CORS preflight.

References: [Render Flask deployment](https://render.com/docs/deploy-flask), [Render monorepo roots](https://render.com/docs/monorepo-support), [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
