# clean_data.py
# Step 9: Structural cleaning of the raw movie data.
# Reads data/raw/movies_metadata.csv. Does NOT change the raw file.
import ast
from pathlib import Path

import pandas as pd

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "movies_metadata.csv"

df = pd.read_csv(RAW_FILE, low_memory=False)
print("Rows at start:", len(df))

# Decision 1: remove corrupted rows (their id is not a number).
id_as_number = pd.to_numeric(df["id"], errors="coerce")
corrupted = id_as_number.isna()
print("\nDecision 1 - corrupted rows removed:", corrupted.sum())
print(df.loc[corrupted, ["id", "budget", "release_date"]])
df = df[~corrupted].copy()
df["id"] = id_as_number[~corrupted].astype("int64")

# Decision 2: remove duplicates.
before = len(df)
df = df.drop_duplicates()
print("\nDecision 2a - identical rows removed:", before - len(df))

before = len(df)
df = df.drop_duplicates(subset="id", keep="first")
print("Decision 2b - extra rows with a repeated id removed:", before - len(df))

# Decision 3: convert data types.
df["budget"] = pd.to_numeric(df["budget"])
df["popularity"] = pd.to_numeric(df["popularity"])
df["release_date"] = pd.to_datetime(df["release_date"])
print("\nDecision 3 - converted types:")
print(df[["id", "budget", "revenue", "popularity", "release_date"]].dtypes)

# Decision 4: keep only movies whose status is 'Released'.
print("\nStatus values before filtering:")
print(df["status"].value_counts(dropna=False))
before = len(df)
df = df[df["status"] == "Released"].copy()
print("\nDecision 4 - rows removed (not released / status missing):", before - len(df))

print("\nRows remaining:", len(df))
print("Earliest release date:", df["release_date"].min())
print("Latest release date:", df["release_date"].max())
print("Movies with no release date:", df["release_date"].isna().sum())
print("Movies with no release date:", ...)


# Decision 5: remove movies with no release date (we cannot place them in time).
before = len(df)
df = df[df["release_date"].notna()].copy()
print("\nDecision 5 - rows removed (no release date):", before - len(df))

# Decision 6: treat impossible money values as 'unknown' (missing).
# A budget or revenue of 0 usually means "not recorded". Tiny values such as 1
# or 12 are not believable for a real movie either. We use 1,000 dollars as a
# simple cut-off and will document it as a judgement call.
MIN_MONEY = 1000
print("\nDecision 6 - money values below", MIN_MONEY, "set to missing:")
print("  budget :", (df["budget"] < MIN_MONEY).sum())
print("  revenue:", (df["revenue"] < MIN_MONEY).sum())
df["budget"] = df["budget"].where(df["budget"] >= MIN_MONEY)
df["revenue"] = df["revenue"].where(df["revenue"] >= MIN_MONEY)

# Mark movies where BOTH numbers are known. We will use these movies for
# budget-versus-revenue analysis and for the prediction model.
df["has_financials"] = df["budget"].notna() & df["revenue"].notna()
print("Movies with both budget and revenue known:", df["has_financials"].sum())

# Decision 7: extract the release year and month from the release date.
df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month
print("\nDecision 7 - release years run from", df["release_year"].min(),
      "to", df["release_year"].max())
print("Movies with known budget and revenue, last 8 release years:")
print(df.loc[df["has_financials"], "release_year"].value_counts().sort_index().tail(8))

print("\nRows remaining:", len(df))


# Decision 8: turn the 'genres' text into usable genre columns.
# In the raw file, genres is TEXT that looks like a Python list, for example
# "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]".
# ast.literal_eval safely turns that text into a real Python list.


def extract_genre_names(text):
    """Return a list of genre names from the raw genres text."""
    items = ast.literal_eval(text)
    return [item["name"] for item in items]


genre_lists = df["genres"].apply(extract_genre_names)
df["genre_count"] = genre_lists.apply(len)
df["primary_genre"] = genre_lists.apply(lambda names: names[0] if names else "Unknown")
df["genres"] = genre_lists.apply(lambda names: "|".join(names))
print("\nDecision 8 - movies with no genre listed:", (df["genre_count"] == 0).sum())
print("Most common primary genres:")
print(df["primary_genre"].value_counts().head(6))

# Decision 9: mark movies that belong to a collection (a series or franchise).
df["in_collection"] = df["belongs_to_collection"].notna()
print("\nDecision 9 - movies that belong to a collection:", df["in_collection"].sum())

# Decision 10: keep only the columns we need and save the cleaned data.
columns_to_keep = [
    "id", "imdb_id", "title", "original_language", "release_date",
    "release_year", "release_month", "runtime", "budget", "revenue",
    "genres", "primary_genre", "genre_count", "in_collection",
    "popularity", "vote_average", "vote_count", "has_financials",
]
clean = df[columns_to_keep].copy()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ALL_FILE = PROCESSED_DIR / "movies_clean.csv"
FINANCIAL_FILE = PROCESSED_DIR / "movies_financial.csv"

clean.to_csv(ALL_FILE, index=False)
clean[clean["has_financials"]].to_csv(FINANCIAL_FILE, index=False)

# Read the saved files back to prove they were written correctly.
check_all = pd.read_csv(ALL_FILE)
check_fin = pd.read_csv(FINANCIAL_FILE)
print("\nDecision 10 - saved files:")
print("  movies_clean.csv    :", check_all.shape)
print("  movies_financial.csv:", check_fin.shape)
print("\nMissing values in movies_financial.csv:")
print(check_fin.isna().sum())