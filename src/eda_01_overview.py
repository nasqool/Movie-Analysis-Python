# eda_01_overview.py
# Step 12: First look at the cleaned data.
#   1. Summary statistics
#   2. The 10 highest-grossing movies
#   3. How recorded revenue changed over the decades
# Charts are saved in outputs/figures. Nothing in data/ is changed.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "movies_financial.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_FILE)
print("Movies with known budget and revenue:", len(df))

# 1. Summary statistics -------------------------------------------------
print("\nSummary statistics (budget and revenue in US dollars):")
summary = df[["budget", "revenue", "runtime"]].describe()
print(summary.round(0).to_string(float_format="{:,.0f}".format))

# 2. Top 10 movies by worldwide revenue -----------------------------------
top10 = df.nlargest(10, "revenue")[["title", "release_year", "budget", "revenue"]]
print("\nTop 10 movies by recorded worldwide revenue:")
print(top10.to_string(index=False, float_format="{:,.0f}".format))

plot_data = top10.sort_values("revenue")  # smallest first, so the biggest ends up on top
plt.figure(figsize=(9, 5))
plt.barh(plot_data["title"], plot_data["revenue"] / 1_000_000, color="#2a6f97")
plt.xlabel("Worldwide revenue (millions of US dollars)")
plt.title("Top 10 movies by recorded worldwide revenue")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_top10_revenue.png", dpi=150)
plt.close()

# 3. Revenue by decade -------------------------------------------------
df["decade"] = (df["release_year"] // 10) * 10
by_decade = df.groupby("decade")["revenue"].agg(movies="count", median_revenue="median")
print("\nBy decade (number of movies and median revenue):")
print(by_decade.to_string(float_format="{:,.0f}".format))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(by_decade.index.astype(str) + "s", by_decade["movies"], color="#8d99ae")
axes[0].set_title("Movies with known budget and revenue")
axes[0].set_ylabel("Number of movies")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(by_decade.index.astype(str) + "s",
            by_decade["median_revenue"] / 1_000_000, color="#2a6f97")
axes[1].set_title("Median worldwide revenue by decade")
axes[1].set_ylabel("Millions of US dollars (not adjusted for inflation)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_revenue_by_decade.png", dpi=150)
plt.close()

print("\nCharts saved in:", FIGURE_DIR)