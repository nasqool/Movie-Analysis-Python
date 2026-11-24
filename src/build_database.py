# build_database.py
# Step 15: Load the cleaned movie data into an SQLite database.
# SQLite is a small database that lives in a single file. It is built into Python.
# Input : data/processed/movies_clean.csv
# Output: data/processed/movies.db  (rebuilt from scratch every time)

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "processed" / "movies_clean.csv"
DB_FILE = PROJECT_ROOT / "data" / "processed" / "movies.db"

# Start fresh so running the script twice never creates duplicate rows.
if DB_FILE.exists():
    DB_FILE.unlink()

movies = pd.read_csv(CSV_FILE)
print("Rows read from movies_clean.csv:", len(movies))

connection = sqlite3.connect(DB_FILE)

# 1. Create the tables. Each column has a name and a data type.
connection.executescript("""
CREATE TABLE movies (
    id                INTEGER PRIMARY KEY,
    imdb_id           TEXT,
    title             TEXT,
    original_language TEXT,
    release_date      TEXT,
    release_year      INTEGER,
    release_month     INTEGER,
    runtime           REAL,
    budget            REAL,
    revenue           REAL,
    genres            TEXT,
    primary_genre     TEXT,
    genre_count       INTEGER,
    in_collection     INTEGER,
    popularity        REAL,
    vote_average      REAL,
    vote_count        REAL,
    has_financials    INTEGER
);

CREATE TABLE movie_genres (
    movie_id INTEGER,
    genre    TEXT,
    FOREIGN KEY (movie_id) REFERENCES movies (id)
);
""")

# 2. Fill the movies table from the DataFrame.
movies.to_sql("movies", connection, if_exists="append", index=False)

# 3. Build the movie_genres table: one row for every (movie, genre) pair.
genre_pairs = movies.dropna(subset=["genres"])[["id", "genres"]].copy()
genre_pairs["genre"] = genre_pairs["genres"].str.split("|")
genre_pairs = genre_pairs.explode("genre")
genre_pairs = genre_pairs.rename(columns={"id": "movie_id"})[["movie_id", "genre"]]
genre_pairs.to_sql("movie_genres", connection, if_exists="append", index=False)

connection.commit()

# 4. Check the result by asking the database a few simple questions.
cursor = connection.cursor()
tables = cursor.execute(
    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
).fetchall()
print("\nTables in the database:", [row[0] for row in tables])

print("Rows in movies         :",
      cursor.execute("SELECT COUNT(*) FROM movies").fetchone()[0])
print("Movies with financials :",
      cursor.execute("SELECT COUNT(*) FROM movies WHERE has_financials = 1").fetchone()[0])
print("Rows in movie_genres   :",
      cursor.execute("SELECT COUNT(*) FROM movie_genres").fetchone()[0])
print("Different genres       :",
      cursor.execute("SELECT COUNT(DISTINCT genre) FROM movie_genres").fetchone()[0])

print("\nFirst 3 movies in the database:")
for row in cursor.execute("SELECT title, release_year, budget, revenue FROM movies LIMIT 3"):
    print(" ", row)

connection.close()
print("\nDatabase saved to:", DB_FILE)