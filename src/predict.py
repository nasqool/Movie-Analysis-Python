# predict.py
# Step 20: A small function that estimates the revenue of ONE new movie
# using the saved model. The dashboard will use this function later.

from pathlib import Path

import joblib
import pandas as pd

from features import make_features

MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "revenue_model.joblib"


def load_model():
    """Load the saved model and its settings from disk."""
    return joblib.load(MODEL_FILE)


def predict_revenue(bundle, budget, runtime, release_month, genres,
                    in_collection=False, is_english=True):
    """Estimate worldwide revenue (US dollars) for one movie.

    Only information known BEFORE release is used.
    Returns the estimate and an approximate 80% range.
    """
    if budget <= 0:
        raise ValueError("The budget must be greater than 0.")
    unknown = [g for g in genres if g not in bundle["genre_list"]]
    if unknown:
        raise ValueError(f"Unknown genre(s): {unknown}")

    # Put the inputs into a one-row table with the same columns as the raw data.
    movie = pd.DataFrame({
        "budget": [budget],
        "runtime": [runtime],
        "release_month": [release_month],
        "in_collection": [bool(in_collection)],
        "original_language": ["en" if is_english else "other"],
        "genres": ["|".join(genres)],
    })
    features = make_features(movie, bundle["genre_list"])
    features = features[bundle["feature_columns"]]  # same column order as in training

    predicted_log = float(bundle["model"].predict(features)[0])
    low_log, high_log = bundle["range_log"]

    notes = []
    low_budget, high_budget = bundle["budget_range"]
    if not low_budget <= budget <= high_budget:
        notes.append("The budget is outside the range seen in the training data, "
                     "so the estimate is less reliable.")

    return {
        "estimate": 10 ** predicted_log,
        "low": 10 ** (predicted_log + low_log),
        "high": 10 ** (predicted_log + high_log),
        "notes": notes,
    }


# Quick check, only when this file is started directly.
if __name__ == "__main__":
    bundle = load_model()
    info = bundle["trained_on"]
    print("Model trained on", info["movies"], "movies released",
          info["first_year"], "to", info["last_year"])

    # An INVENTED example movie, only to show how the function is used.
    result = predict_revenue(
        bundle,
        budget=50_000_000,
        runtime=110,
        release_month=6,
        genres=["Action", "Adventure"],
        in_collection=False,
        is_english=True,
    )
    print("\nInvented example: budget $50M, 110 min, June release, Action + Adventure")
    print(f"  Estimate  : ${result['estimate']:,.0f}")
    print(f"  80% range : ${result['low']:,.0f}  to  ${result['high']:,.0f}")