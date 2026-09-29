# Housing Predictor

Initial configurable linear regression training pipeline. Run these commands from
the project folder using Python 3.10 or newer:

```sh
python -m pip install -r backend/requirements.txt
python backend/train_model.py
```

The script locates its configuration and dataset relative to its own file, so it
also works when launched from another directory. Training uses `data/housing.csv`
and saves the complete fitted preprocessing and regression pipeline to
`backend/model.pkl` after training and evaluation succeed.

Edit `backend/model_config.json` to change the target or predictors. Put numeric
columns in `numeric_features` and category columns (such as neighborhood or
quality labels) in `categorical_features`, then retrain. Every configured column
must exist in the CSV; the target must be numeric and cannot be a predictor.

Numeric missing values use training-set medians. Categorical missing values use
the most frequent training category before one-hot encoding; unseen categories
are accepted during prediction. All-missing training columns are retained with a
zero fallback and reported. Preprocessing is fitted only on the training data.
The reproducible 80/20 split reports holdout R², RMSE, and MAE; the saved pipeline
is the same model evaluated on that holdout.

Rows with missing targets are removed. Blank cells and standard pandas CSV NA
markers count as missing. Nonnumeric numeric values and infinities produce clear
errors. Negative values (including possible survey codes such as -6 and -9) are
reported and retained: consult the dataset codebook and replace confirmed missing
codes with blank cells before interpreting model performance. The CSV is never
modified by the script.

For later prediction, load the artifact with `joblib.load` and call `predict`
with a pandas DataFrame containing all configured predictors in their original
units/categories. Use the same dependency versions for training and loading.
The Flask endpoint and frontend are not implemented yet.
