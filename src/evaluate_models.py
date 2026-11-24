# evaluate_models.py
# Step 19: Look more closely at the models before choosing one.
#   1. Repeat the test with different cut-off years (is the result stable?)
#   2. Where do the predictions go wrong? (by budget, biggest misses)
#   3. Which features does the Random Forest rely on most?
# Nothing in data/ is changed.

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

from features import (
    LAST_TRAIN_YEAR,
    get_genre_list,
    load_financial_movies,
    make_features,
    make_target,
)

movies = load_financial_movies()
genre_list = get_genre_list(movies)
X = make_features(movies, genre_list)
y = make_target(movies)


def make_models():
    """Create fresh, untrained copies of the three models."""
    return {
        "Baseline (median)": DummyRegressor(strategy="median"),
        "Linear Regression": make_pipeline(SimpleImputer(strategy="median"), LinearRegression()),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
        ),
    }


# 1. Different cut-off years -------------------------------------------------
# For each cut-off we train on movies up to that year and test on the NEXT 3 years.
print("1. Is the result stable? (train up to the cut-off year, test on the next 3 years)\n")
rows = []
for cutoff in [2009, 2011, 2013]:
    train = movies["release_year"] <= cutoff
    test = (movies["release_year"] > cutoff) & (movies["release_year"] <= cutoff + 3)
    for name, model in make_models().items():
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        rows.append({
            "train up to": cutoff,
            "test years": f"{cutoff + 1}-{cutoff + 3}",
            "test movies": int(test.sum()),
            "model": name,
            "R2 (log)": r2_score(y[test], pred),
            "typical factor": 10 ** mean_absolute_error(y[test], pred),
            "MAE ($M)": mean_absolute_error(10 ** y[test], 10 ** pred) / 1e6,
        })
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 2. Error analysis for the Random Forest on the main split ----------------
train = movies["release_year"] <= LAST_TRAIN_YEAR
test = ~train
forest = make_models()["Random Forest"].fit(X[train], y[train])

report = movies.loc[test, ["title", "release_year", "budget", "revenue"]].copy()
report["predicted"] = 10 ** forest.predict(X[test])
report["error_log"] = forest.predict(X[test]) - y[test]  # positive = predicted too high
report["abs_error_log"] = report["error_log"].abs()

print("\n\n2a. Accuracy by budget group (Random Forest, test movies)\n")
report["budget group"] = pd.qcut(
    report["budget"], 4,
    labels=["lowest 25%", "25-50%", "50-75%", "highest 25%"],
)
by_group = report.groupby("budget group", observed=True).agg(
    movies=("title", "count"),
    typical_factor=("abs_error_log", lambda s: 10 ** s.mean()),
    share_too_high=("error_log", lambda s: (s > 0).mean() * 100),
)
print(by_group.round(2).to_string())

show = ["title", "release_year", "budget", "revenue", "predicted"]
money = {"budget": "{:,.0f}".format, "revenue": "{:,.0f}".format, "predicted": "{:,.0f}".format}

print("\n2b. Predicted much HIGHER than the actual revenue (top 8)\n")
print(report.nlargest(8, "error_log")[show].to_string(index=False, formatters=money))

print("\n2c. Predicted much LOWER than the actual revenue (top 8)\n")
print(report.nsmallest(8, "error_log")[show].to_string(index=False, formatters=money))

# 3. Which features matter most to the Random Forest? ----------------------
importance = pd.Series(
    forest.named_steps["randomforestregressor"].feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n\n3. Most used features in the Random Forest (shares add up to 1)\n")
print(importance.head(8).round(3).to_string())
