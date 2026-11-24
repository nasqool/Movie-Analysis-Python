# features.py
# Step 17: Turn the cleaned movie table into model inputs (features) and a target.
#
# IMPORTANT: a feature may only use information that is known BEFORE a movie is
# released. Using information from after the release is called "data leakage":
# the model would look accurate on paper but could not be used in real life.
#
# Features we use (all known before release):
#   budget (as log10), runtime, release month, part of a collection, English
#   language, and the genres.
# Columns we deliberately do NOT use:
#   popularity, vote_average, vote_count  -> collected after release
#   release_year                          -> see the note in make_features()
#   id, imdb_id, title                    -> labels, not information about success

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "movies_financial.csv"

# Movies released up to this year are used for training.
# Movies released later are used only for testing, to simulate predicting the future.
LAST_TRAIN_YEAR = 2013


def load_financial_movies():
    """Read the cleaned movies that have a known budget and revenue."""
    return pd.read_csv(DATA_FILE)


def get_genre_list(df):
    """Return every genre that appears in the data, in alphabetical order."""
    genres = df["genres"].dropna().str.split("|").explode()
    return sorted(genres.unique())


def make_features(df, genre_list):
    """Build the table of model inputs (X) from a table of movies."""
    features = pd.DataFrame(index=df.index)

    # Budgets range from thousands to hundreds of millions of dollars,
    # so we use the logarithm (log10) of the budget.
    features["log_budget"] = np.log10(df["budget"])

    # A runtime of 0 minutes is not real, so we treat it as missing.
    features["runtime"] = df["runtime"].where(df["runtime"] > 0)

    features["release_month"] = df["release_month"]
    features["in_collection"] = df["in_collection"].astype(int)
    features["is_english"] = (df["original_language"] == "en").astype(int)

    # One 0/1 column for each genre ("one-hot" style encoding).
    genre_lists = df["genres"].fillna("").str.split("|")
    for genre in genre_list:
        column_name = "genre_" + genre.replace(" ", "_")
        features[column_name] = genre_lists.apply(lambda names: int(genre in names))

    # Note: we leave out release_year. Because we test on LATER years than we
    # train on, the model would see year values it has never seen before.
    return features


def make_target(df):
    """The value we want to predict: log10 of worldwide revenue."""
    return np.log10(df["revenue"])


def split_by_time(df):
    """Return True/False masks that pick the training movies and the test movies."""
    train_mask = df["release_year"] <= LAST_TRAIN_YEAR
    test_mask = ~train_mask
    return train_mask, test_mask


# This block runs only when we start THIS file directly. It is a quick check.
if __name__ == "__main__":
    movies = load_financial_movies()
    genre_list = get_genre_list(movies)
    X = make_features(movies, genre_list)
    y = make_target(movies)
    train_mask, test_mask = split_by_time(movies)

    print("Movies:", len(movies))
    print("Number of features:", X.shape[1])
    print("Feature columns:", list(X.columns))

    print("\nTraining movies:", train_mask.sum(),
          "(released", movies.loc[train_mask, "release_year"].min(),
          "to", movies.loc[train_mask, "release_year"].max(), ")")
    print("Test movies    :", test_mask.sum(),
          "(released", movies.loc[test_mask, "release_year"].min(),
          "to", movies.loc[test_mask, "release_year"].max(), ")")

    print("\nMissing values in the features (only columns with a gap):")
    missing = X.isna().sum()
    print(missing[missing > 0])

    print("\nTarget (log10 of revenue) - first 3 values:")
    print(y.head(3).round(3).to_string())
    print("Same movies in dollars:")
    print(movies["revenue"].head(3).round(0).to_string())