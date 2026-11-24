# eda_03_roi_timing_ratings.py
# Step 14: Return relative to budget, release timing and audience ratings.
# Charts are saved in outputs/figures. Nothing in data/ is changed.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "movies_financial.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_FILE)
print("Movies with known budget and revenue:", len(df))

# 1. Revenue relative to budget -------------------------------------------
# revenue_multiple = worldwide revenue divided by production budget.
# This is NOT profit: cinemas keep part of ticket sales, and marketing and
# other costs are not in the budget figure.
df["revenue_multiple"] = df["revenue"] / df["budget"]

print("\nRevenue multiple (revenue / budget), all movies:")
print("  median:", round(df["revenue_multiple"].median(), 2))
print("  movies whose revenue is below their budget:",
      (df["revenue_multiple"] < 1).sum(),
      "(", round((df["revenue_multiple"] < 1).mean() * 100, 1), "% )")

# Very small budgets give extreme multiples, so for the ranking we only
# use movies with a budget of at least 1 million dollars.
MIN_BUDGET_FOR_RANKING = 1_000_000
ranked = df[df["budget"] >= MIN_BUDGET_FOR_RANKING]
top_multiple = ranked.nlargest(10, "revenue_multiple")[
    ["title", "release_year", "budget", "revenue", "revenue_multiple"]
]
print("\nTop 10 revenue multiples (budget of at least 1 million dollars):")
print(top_multiple.to_string(index=False, float_format="{:,.1f}".format))

plot_data = top_multiple.sort_values("revenue_multiple")
plt.figure(figsize=(9, 5))
plt.barh(plot_data["title"], plot_data["revenue_multiple"], color="#2a6f97")
plt.xlabel("Worldwide revenue divided by budget")
plt.title("Highest revenue multiples (budget of at least $1 million)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_top_revenue_multiples.png", dpi=150)
plt.close()

# 2. Release month -------------------------------------------------------
by_month = df.groupby("release_month")["revenue"].agg(
    movies="count", median_revenue="median", mean_revenue="mean"
)
print("\nRevenue by release month:")
print(by_month.to_string(float_format="{:,.0f}".format))

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.figure(figsize=(9, 4.5))
plt.bar(month_names, by_month["median_revenue"] / 1_000_000, color="#2a6f97")
plt.ylabel("Median worldwide revenue (millions of US dollars)")
plt.title("Median revenue by release month")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "06_revenue_by_month.png", dpi=150)
plt.close()

# 3. Audience ratings versus revenue -------------------------------------------
# vote_average and vote_count are collected AFTER release, so we use them
# here only to describe the data. They must NOT be used to predict revenue
# before a movie is released.
MIN_VOTES = 50  # an average from only a few votes is not reliable
rated = df[df["vote_count"] >= MIN_VOTES]
print("\nMovies with at least", MIN_VOTES, "votes:", len(rated))
print("Correlation (Spearman) between rating and revenue:",
      round(rated["vote_average"].corr(rated["revenue"], method="spearman"), 3))
print("Correlation (Spearman) between number of votes and revenue:",
      round(df["vote_count"].corr(df["revenue"], method="spearman"), 3))

plt.figure(figsize=(8, 6))
plt.scatter(rated["vote_average"], rated["revenue"], s=8, alpha=0.35, color="#2a6f97")
plt.yscale("log")
plt.xlabel("Average audience rating (out of 10)")
plt.ylabel("Worldwide revenue (US dollars, log scale)")
plt.title("Audience rating versus worldwide revenue")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "07_rating_vs_revenue.png", dpi=150)
plt.close()

print("\nCharts saved in:", FIGURE_DIR)