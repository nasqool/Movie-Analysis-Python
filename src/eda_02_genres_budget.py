# eda_02_genres_budget.py
# Step 13: Genres, and the relationship between budget and revenue.
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

# 1. Revenue by genre ---------------------------------------------------
# A movie can have several genres, e.g. "Animation|Comedy|Family".
# We split that text into separate genres and give the movie one row per genre,
# so a movie counts once in every genre it belongs to.
genre_rows = df.dropna(subset=["genres"]).copy()
genre_rows["genre"] = genre_rows["genres"].str.split("|")
genre_rows = genre_rows.explode("genre")

MIN_MOVIES = 30  # ignore genres with too few movies to be trustworthy
by_genre = genre_rows.groupby("genre")["revenue"].agg(
    movies="count", mean_revenue="mean", median_revenue="median"
)
by_genre = by_genre[by_genre["movies"] >= MIN_MOVIES].sort_values(
    "mean_revenue", ascending=False
)

print("\nRevenue by genre (genres with at least", MIN_MOVIES, "movies):")
print(by_genre.to_string(float_format="{:,.0f}".format))

plot_data = by_genre.sort_values("mean_revenue")
positions = np.arange(len(plot_data))
plt.figure(figsize=(9, 6))
plt.barh(positions + 0.2, plot_data["mean_revenue"] / 1_000_000,
         height=0.4, color="#2a6f97", label="Average (mean)")
plt.barh(positions - 0.2, plot_data["median_revenue"] / 1_000_000,
         height=0.4, color="#8d99ae", label="Median")
plt.yticks(positions, plot_data.index)
plt.xlabel("Worldwide revenue (millions of US dollars)")
plt.title("Revenue by genre (movies with known budget and revenue)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "03_revenue_by_genre.png", dpi=150)
plt.close()

# 2. Budget versus revenue -----------------------------------------------
print("\nCorrelation between budget and revenue:")
pearson = df["budget"].corr(df["revenue"])
spearman = df["budget"].corr(df["revenue"], method="spearman")
log_pearson = np.log10(df["budget"]).corr(np.log10(df["revenue"]))
print("  Pearson (raw dollars)      :", round(pearson, 3))
print("  Spearman (ranks)           :", round(spearman, 3))
print("  Pearson (log of both)      :", round(log_pearson, 3))

plt.figure(figsize=(8, 6))
plt.scatter(df["budget"], df["revenue"], s=8, alpha=0.35, color="#2a6f97")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Budget (US dollars, log scale)")
plt.ylabel("Worldwide revenue (US dollars, log scale)")
plt.title("Budget versus worldwide revenue")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_budget_vs_revenue.png", dpi=150)
plt.close()

print("\nCharts saved in:", FIGURE_DIR)