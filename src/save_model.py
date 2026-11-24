# save_model.py
# Step 20: Train the FINAL model and save it to a file with Joblib.
#
#   1. Re-run the test (train up to 2013, test 2014-2017) to measure the model
#      and to find a typical error range for its predictions.
#   2. Train the final Random Forest on ALL movies (1915-2017), so that it has
#      learned from as much history as possible.
#   3. Save the model plus everything needed to use it later in
#      models/revenue_model.joblib

import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

from features import (
    get_genre_list,
    load_financial_movies,
    make_features,
    make_target,
    split_by_time,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "revenue_model.joblib"


def new_forest():
    """A fresh, untrained Random Forest (same settings as in Step 18)."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
    )


movies = load_financial_movies()
genre_list = get_genre_list(movies)
X = make_features(movies, genre_list)
y = make_target(movies)
train_mask, test_mask = split_by_time(movies)

# 1. Test the model on movies it has never seen ----------------------------
test_model = new_forest().fit(X[train_mask], y[train_mask])
test_pred = test_model.predict(X[test_mask])
residual = y[test_mask] - test_pred  # actual minus predicted (log10 scale)

test_metrics = {
    "r2_log": float(r2_score(y[test_mask], test_pred)),
    "typical_factor": float(10 ** mean_absolute_error(y[test_mask], test_pred)),
    "mae_dollars": float(mean_absolute_error(10 ** y[test_mask], 10 ** test_pred)),
    "test_movies": int(test_mask.sum()),
}

# The middle 80% of the test errors: the range below was wide enough to contain
# the actual revenue for 80% of the 2014-2017 test movies.
low_log, high_log = np.percentile(residual, [10, 90])
print("Test result (train up to 2013, test 2014-2017):")
print("  R2 on log scale      :", round(test_metrics["r2_log"], 3))
print("  typical error factor :", round(test_metrics["typical_factor"], 2))
print("  MAE in dollars       :", f"{test_metrics['mae_dollars']:,.0f}")
print("  80% range multipliers: x", round(10 ** low_log, 2), "to x", round(10 ** high_log, 2))

# 2. Train the final model on ALL movies ------------------------------------
final_model = new_forest().fit(X, y)

# 3. Save everything needed to use the model later ------------------------
bundle = {
    "model": final_model,
    "genre_list": genre_list,
    "feature_columns": list(X.columns),
    "range_log": (float(low_log), float(high_log)),
    "budget_range": (float(movies["budget"].min()), float(movies["budget"].max())),
    "trained_on": {
        "movies": int(len(movies)),
        "first_year": int(movies["release_year"].min()),
        "last_year": int(movies["release_year"].max()),
    },
    "test_metrics": test_metrics,
    "created": datetime.date.today().isoformat(),
    "scikit_learn_version": sklearn.__version__,
}
joblib.dump(bundle, MODEL_FILE, compress=3)  # compress=3 makes the file much smaller
print("\nModel saved to:", MODEL_FILE)
print("File size (KB):", round(MODEL_FILE.stat().st_size / 1024))

# 4. Prove that the saved file works: load it and compare predictions --------
loaded = joblib.load(MODEL_FILE)
same = np.allclose(loaded["model"].predict(X.head(200)), final_model.predict(X.head(200)))
print("Loaded model gives the same predictions as the original:", same)