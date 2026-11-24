# run_sql.py
# Step 16: Run every query in a .sql file against the movie database
# and print each result as a table.
#
# Usage (from the project folder):
#   .\.venv\Scripts\python.exe src\run_sql.py sql\01_financial_queries.sql

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_ROOT / "data" / "processed" / "movies.db"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

if len(sys.argv) != 2:
    print("Please give the path of one .sql file, for example:")
    print(r"  .\.venv\Scripts\python.exe src\run_sql.py sql\01_financial_queries.sql")
    sys.exit(1)

sql_file = Path(sys.argv[1])
if not sql_file.is_absolute():
    sql_file = PROJECT_ROOT / sql_file
sql_text = sql_file.read_text(encoding="utf-8")

connection = sqlite3.connect(DB_FILE)

# Queries in the file are separated by semicolons.
for block in sql_text.split(";"):
    lines = [line for line in block.strip().splitlines() if line.strip()]
    if not lines:
        continue

    # Lines starting with "--" are comments. The comment that begins with
    # "Q1:", "Q2:" ... is the query's title. Comment lines after the title are
    # printed as notes.
    comments = [line for line in lines if line.startswith("--")]
    query = "\n".join(line for line in lines if not line.startswith("--"))
    if not query.strip():
        continue

    title_lines = [c for c in comments if re.match(r"--\s*Q\d+", c)]
    title = title_lines[0].lstrip("- ").strip() if title_lines else "Query"
    notes = []
    if title_lines:
        notes = comments[comments.index(title_lines[0]) + 1:]

    print("\n" + "=" * 100)
    print(title)
    for note in notes:
        print("   " + note.lstrip("- ").strip())
    print("=" * 100)
    result = pd.read_sql_query(query, connection)
    print(result.to_string(index=False, float_format="{:,.1f}".format))

connection.close()