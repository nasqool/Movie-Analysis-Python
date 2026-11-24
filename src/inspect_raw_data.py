# inspect_raw_data.py
# Step 8: Look at the raw dataset BEFORE cleaning it.
# This script only reads and prints. It does not change any file.

from pathlib import Path

import pandas as pd

# Make wide tables easier to read in the terminal.
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

# 1. Build the path to the CSV file, wherever we run the script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "movies_metadata.csv"

# 2. Load the CSV into a DataFrame (a table with rows and columns).
df = pd.read_csv(RAW_FILE, low_memory=False)

# 3. How big is the table?
print("Rows, columns:", df.shape)

# 4. Column names and the data type Pandas chose for each.
print("\nColumns and data types:")
print(df.dtypes)

# 5. How many values are missing in each column?
print("\nMissing values per column:")
print(df.isna().sum())

# 6. Duplicates.
print("\nFully duplicated rows:", df.duplicated().sum())
print("Repeated movie ids:", df["id"].duplicated().sum())

# 7. A first look at a few important columns.
print("\nFirst 3 rows of key columns:")
print(df[["title", "budget", "revenue", "release_date", "genres"]].head(3))