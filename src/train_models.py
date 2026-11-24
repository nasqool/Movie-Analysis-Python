# train_models.py
# Step 18: Train and compare models that predict worldwide revenue.
#
#   1. Baseline          : always predicts the median revenue of the training movies
#   2. Linear (budget)   : Linear Regression using only the budget
#   3. Linear Regression : all pre-release features
#   4. Random Forest     : all pre-release features
#
# Training movies: released up to 2013. Test movies: released 2014 to 2017.
# The models predict log10(revenue); results are shown on that scale AND in dollars.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

from features import (
    get_genre_list,
    load_financial_movies,
    make_features,
    make_target,
    split_by_time,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Prepare the data --------------------------------------------------------
movies = load_financial_movies()
genre_list = get_genre_list(movies)
X = make_features(movies, genre_list)
y = make_target(movies)
train_mask, test_mask = split_by_time(movies)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
print("Training movies:", len(X_train), "| Test movies:", len(X_test))

# 2. Define the models -------------------------------------------------------
# SimpleImputer fills the few missing runtimes with the MEDIAN runtime of the
# TRAINING movies only (so no information leaks in from the test movies).
models = {
    "Baseline (median)": (DummyRegressor(strategy="median"), list(X.columns)),
    "Linear (budget only)": (make_pipeline(LinearRegression()), ["log_budget"]),
    "Linear Regression": (
        make_pipeline(SimpleImputer(strategy="median"), LinearRegression()),
        list(X.columns),
    ),
    "Random Forest": (
        make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
        ),
        list(X.columns),
    ),
}

# 3. Train each model and measure it on the test movies --------------------
rows = []
test_predictions = {}
for name, (model, columns) in models.items():
    model.fit(X_train[columns], y_train)
    pred_train = model.predict(X_train[columns])
    pred_test = model.predict(X_test[columns])
    test_predictions[name] = pred_test

    # Errors in dollars: convert log10 values back with 10 ** value.
    actual_dollars = 10 ** y_test
    predicted_dollars = 10 ** pred_test

    mae_log = mean_absolute_error(y_test, pred_test)
    rows.append({
        "model": name,
        "MAE ($M)": mean_absolute_error(actual_dollars, predicted_dollars) / 1e6,
        "RMSE ($M)": mean_squared_error(actual_dollars, predicted_dollars) ** 0.5 / 1e6,
        "R2 ($)": r2_score(actual_dollars, predicted_dollars),
        "MAE (log)": mae_log,
        "RMSE (log)": mean_squared_error(y_test, pred_test) ** 0.5,
        "R2 (log)": r2_score(y_test, pred_test),
        "typical factor": 10 ** mae_log,
        "R2 (log) on training": r2_score(y_train, pred_train),
    })

results = pd.DataFrame(rows).set_index("model")
print("\nResults on the TEST movies (released 2014 to 2017):")
print(results.round(3).to_string())

results.round(4).to_csv(OUTPUT_DIR / "model_results.csv")
print("\nResults table saved to:", OUTPUT_DIR / "model_results.csv")

# 4. Chart: predicted versus actual revenue ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
low, high = float(y_test.min()) - 0.2, float(y_test.max()) + 0.2
for ax, name in zip(axes, ["Linear Regression", "Random Forest"]):
    ax.scatter(10 ** y_test, 10 ** test_predictions[name], s=10, alpha=0.4, color="#2a6f97")
    ax.plot([10 ** low, 10 ** high], [10 ** low, 10 ** high], color="#c1121f", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(name)
    ax.set_xlabel("Actual revenue (US dollars, log scale)")
axes[0].set_ylabel("Predicted revenue (US dollars, log scale)")
fig.suptitle("Predicted versus actual revenue, test movies (red line = perfect prediction)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "08_predicted_vs_actual.png", dpi=150)
plt.close()
print("Chart saved to:", FIGURE_DIR / "08_predicted_vs_actual.png")