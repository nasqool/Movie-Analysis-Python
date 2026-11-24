# app.py
# The Streamlit dashboard for the Movie Box Office Analysis project.
# Start it from the project folder with:
#   .\.venv\Scripts\python.exe -m streamlit run app.py

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Allow charts with more than 5,000 rows (our scatter chart has about 5,300 points).
alt.data_transformers.disable_max_rows()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "movies_financial.csv"

sys.path.insert(0, str(PROJECT_ROOT / "src"))  # lets Python find predict.py in the src folder
from predict import load_model, predict_revenue  # noqa: E402

st.set_page_config(page_title="Movie Box Office Analysis", layout="wide")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Read the cleaned data once and keep it in memory (cached)."""
    return pd.read_csv(DATA_FILE)


def money_short(value):
    """Write a dollar amount in a short, readable form, e.g. $91.6M."""
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value:,.0f}"


def money_md(value):
    """Like money_short, but safe inside Streamlit text (a pair of dollar signs would start a formula)."""
    return money_short(value).replace("$", "\\$")


# ---------------------------------------------------------------------------
# Load the data and build the sidebar filters
# ---------------------------------------------------------------------------
df = load_data()
all_genres = sorted(df["genres"].dropna().str.split("|").explode().unique())

st.sidebar.header("Filters")
selected_genres = st.sidebar.multiselect(
    "Genres",
    options=all_genres,
    default=[],
    help="Leave empty to include every genre. A movie is kept if it has ANY of the chosen genres.",
)
first_year, last_year = int(df["release_year"].min()), int(df["release_year"].max())
years = st.sidebar.slider("Release years", first_year, last_year, (first_year, last_year))

# Keep only the movies that match the filters.
keep = df["release_year"].between(years[0], years[1])
if selected_genres:
    genre_lists = df["genres"].fillna("").str.split("|")
    keep = keep & genre_lists.apply(lambda names: any(g in names for g in selected_genres))
filtered = df[keep]

# ---------------------------------------------------------------------------
# Page title and tabs
# ---------------------------------------------------------------------------
st.title("Movie Box Office Analysis")
st.caption(
    "Based on movies with a known budget and revenue (from The Movies Dataset on Kaggle, "
    "data up to 2017). Dollar amounts are as recorded in the dataset and are not adjusted "
    "for inflation. Worldwide revenue is gross revenue, not profit."
)

if filtered.empty:
    st.warning("No movies match these filters. Try widening the years or removing a genre.")
    st.stop()

tab_overview, tab_budget, tab_trends, tab_top, tab_predict = st.tabs(
    ["Overview", "Budget vs revenue", "Yearly trends", "Top movies", "Predict revenue"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Industry overview")

    best = filtered.loc[filtered["revenue"].idxmax()]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies", f"{len(filtered):,}")
    col2.metric("Median budget", money_short(filtered["budget"].median()))
    col3.metric("Median revenue", money_short(filtered["revenue"].median()))
    col4.metric("Highest revenue", money_short(best["revenue"]))
    st.caption(f"Highest revenue in this selection: {best['title']} ({best['release_year']})")

    # --- Revenue by genre -------------------------------------------------
    st.subheader("Revenue by genre")
    measure = st.radio(
        "Show",
        ["Average (mean)", "Median"],
        horizontal=True,
        help="A few huge hits pull the average up, so the median is often a better 'typical' value.",
    )

    genre_rows = filtered.dropna(subset=["genres"]).copy()
    genre_rows["Genre"] = genre_rows["genres"].str.split("|")
    genre_rows = genre_rows.explode("Genre")

    by_genre = genre_rows.groupby("Genre")["revenue"].agg(["count", "mean", "median"])
    by_genre = by_genre[by_genre["count"] >= 10]  # skip genres with very few movies

    if by_genre.empty:
        st.info("Not enough movies per genre for this selection (at least 10 are needed).")
    else:
        value_column = "mean" if measure == "Average (mean)" else "median"
        chart_data = pd.DataFrame({
            "Genre": by_genre.index,
            "revenue_m": (by_genre[value_column] / 1_000_000).round(1).values,
        })
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "revenue_m:Q",
                    title="Revenue (millions of US dollars)",
                    scale=alt.Scale(domain=[0, float(chart_data["revenue_m"].max()) * 1.05]),
                ),
                y=alt.Y("Genre:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("Genre:N"),
                    alt.Tooltip("revenue_m:Q", title="Revenue ($M)", format=",.1f"),
                ],
            )
        )
        st.altair_chart(chart)
        st.caption(
            "A movie counts once in every genre it belongs to, so genres overlap. "
            "Genres with fewer than 10 movies in the selection are left out."
        )

    # --- Top 5 movies -------------------------------------------------------
    st.subheader("Top 5 movies by worldwide revenue")
    top5 = filtered.nlargest(5, "revenue")[["title", "release_year", "budget", "revenue"]].copy()
    top5["budget"] = top5["budget"].map(money_short)
    top5["revenue"] = top5["revenue"].map(money_short)
    top5.columns = ["Title", "Year", "Budget", "Worldwide revenue"]
    st.dataframe(top5, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: Budget vs revenue
# ---------------------------------------------------------------------------
with tab_budget:
    st.subheader("Budget versus revenue")

    use_log = st.checkbox(
        "Use logarithmic axes",
        value=True,
        help="Budgets and revenues range from thousands to billions of dollars. "
             "On a log scale each gridline is 10 times the previous one, so small and "
             "big movies fit on one chart.",
    )

    scatter_data = filtered[["title", "release_year", "budget", "revenue"]].copy()
    scatter_data["budget_m"] = scatter_data["budget"] / 1_000_000
    scatter_data["revenue_m"] = scatter_data["revenue"] / 1_000_000
    axis_type = "log" if use_log else "linear"

    scatter = (
        alt.Chart(scatter_data)
        .mark_circle(size=45, opacity=0.45)
        .encode(
            x=alt.X("budget_m:Q", title="Budget (millions of US dollars)",
                    scale=alt.Scale(type=axis_type)),
            y=alt.Y("revenue_m:Q", title="Worldwide revenue (millions of US dollars)",
                    scale=alt.Scale(type=axis_type)),
            tooltip=[
                alt.Tooltip("title:N", title="Title"),
                alt.Tooltip("release_year:Q", title="Year", format="d"),
                alt.Tooltip("budget_m:Q", title="Budget ($M)", format=",.1f"),
                alt.Tooltip("revenue_m:Q", title="Revenue ($M)", format=",.1f"),
            ],
        )
    )
    st.altair_chart(scatter)
    st.caption("Each dot is one movie. Hover over a dot to see the movie, its budget and its revenue.")

    colA, colB = st.columns(2)
    if len(filtered) >= 10:
        rank_corr = filtered["budget"].corr(filtered["revenue"], method="spearman")
        colA.metric("Correlation (budget and revenue)", f"{rank_corr:.2f}",
                    help="Spearman rank correlation from -1 to +1. Values near +1 mean that "
                         "movies with bigger budgets tend to have bigger revenues.")
    else:
        colA.metric("Correlation (budget and revenue)", "n/a",
                    help="At least 10 movies are needed.")
    below = (filtered["revenue"] < filtered["budget"]).mean() * 100
    colB.metric("Movies with revenue below budget", f"{below:.1f}%",
                help="Gross revenue below budget does not automatically mean a financial loss.")
    st.caption(
        "A strong correlation does not prove that a bigger budget causes a bigger revenue. "
        "Studios usually spend more on movies they already expect to do well."
    )

# ---------------------------------------------------------------------------
# Tab 3: Yearly trends
# ---------------------------------------------------------------------------
with tab_trends:
    st.subheader("Yearly trends")

    trend_measure = st.radio(
        "Show",
        ["Median revenue", "Average revenue", "Total revenue", "Number of movies"],
        horizontal=True,
        key="trend_measure",
    )
    hide_small_years = st.checkbox("Hide years with fewer than 5 movies", value=True)

    by_year = filtered.groupby("release_year")["revenue"].agg(
        movies="count", median="median", mean="mean", total="sum"
    ).reset_index()
    if hide_small_years:
        by_year = by_year[by_year["movies"] >= 5]

    if by_year.empty:
        st.info("Not enough movies in this selection to draw a trend.")
    else:
        if trend_measure == "Number of movies":
            by_year["value"] = by_year["movies"]
            y_title = "Number of movies with known budget and revenue"
        else:
            column = {"Median revenue": "median", "Average revenue": "mean",
                      "Total revenue": "total"}[trend_measure]
            by_year["value"] = by_year[column] / 1_000_000
            y_title = trend_measure + " (millions of US dollars)"

        trend = (
            alt.Chart(by_year)
            .mark_line(point=True)
            .encode(
                x=alt.X("release_year:Q", title="Release year",
                        scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
                y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("release_year:Q", title="Year", format="d"),
                    alt.Tooltip("value:Q", title=trend_measure, format=",.1f"),
                    alt.Tooltip("movies:Q", title="Movies in the data"),
                ],
            )
        )
        st.altair_chart(trend)
        st.caption(
            "Only movies with a recorded budget and revenue are counted, and dollars are not "
            "adjusted for inflation, so these lines show what the dataset recorded, not "
            "the true growth of the film industry. Totals depend on how many movies are "
            "recorded each year. The data ends partway through 2017."
        )

# ---------------------------------------------------------------------------
# Tab 4: Top movies
# ---------------------------------------------------------------------------
with tab_top:
    st.subheader("Top-performing movies")

    left, right = st.columns(2)
    rank_by = left.selectbox(
        "Rank by", ["Worldwide revenue", "Revenue multiple (revenue / budget)"]
    )
    how_many = right.slider("Number of movies", 5, 50, 10)

    ranked = filtered.copy()
    ranked["revenue_multiple"] = ranked["revenue"] / ranked["budget"]
    if rank_by.startswith("Revenue multiple"):
        ranked = ranked[ranked["budget"] >= 1_000_000]
        sort_column = "revenue_multiple"
        st.caption(
            "Only movies with a budget of at least $1 million are ranked here, because tiny "
            "budgets give extreme multiples. The revenue multiple is gross revenue divided "
            "by budget. It is not profit."
        )
    else:
        sort_column = "revenue"

    if ranked.empty:
        st.info("No movies match this selection.")
    else:
        result = ranked.nlargest(how_many, sort_column)[
            ["title", "release_year", "primary_genre", "budget", "revenue", "revenue_multiple"]
        ].copy()
        result.insert(0, "rank", range(1, len(result) + 1))

        shown = result.copy()
        shown["budget"] = shown["budget"].map(money_short)
        shown["revenue"] = shown["revenue"].map(money_short)
        shown["revenue_multiple"] = shown["revenue_multiple"].round(1)
        shown.columns = ["Rank", "Title", "Year", "First-listed genre", "Budget",
                         "Worldwide revenue", "Revenue multiple"]
        st.dataframe(shown, hide_index=True)

        st.download_button(
            "Download this table as CSV",
            data=result.to_csv(index=False),
            file_name="top_movies.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Tab 5: Predict revenue (built in a later step)
# ---------------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


@st.cache_resource
def get_model():
    """Load the saved model once and keep it in memory."""
    return load_model()


with tab_predict:
    st.subheader("Estimate the revenue of a movie before its release")
    st.write(
        "Enter what is known about a movie BEFORE it comes out. The model, trained on "
        "historical movies, gives a rough estimate of its worldwide revenue. "
        "It is a statistical estimate, not a promise."
    )

    try:
        bundle = get_model()
    except FileNotFoundError:
        st.error("The saved model was not found. Run src\\save_model.py first (Step 20).")
        bundle = None

    if bundle is not None:
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            budget = col_a.number_input(
                "Budget (US dollars)", min_value=1_000, max_value=1_000_000_000,
                value=50_000_000, step=1_000_000,
            )
            runtime = col_b.number_input(
                "Runtime (minutes)", min_value=40, max_value=300, value=110, step=5,
            )
            month_name = col_a.selectbox("Release month", MONTHS, index=5)
            chosen_genres = col_b.multiselect(
                "Genres", bundle["genre_list"], default=["Action"],
            )
            in_collection = col_a.checkbox("Part of a series or franchise")
            is_english = col_b.checkbox("Original language is English", value=True)
            submitted = st.form_submit_button("Estimate revenue")

        if submitted:
            if not chosen_genres:
                st.warning("Please choose at least one genre.")
            else:
                result = predict_revenue(
                    bundle,
                    budget=budget,
                    runtime=runtime,
                    release_month=MONTHS.index(month_name) + 1,
                    genres=chosen_genres,
                    in_collection=in_collection,
                    is_english=is_english,
                )
                st.metric("Estimated worldwide revenue", money_short(result["estimate"]))
                st.write(
                    f"Likely range: **{money_md(result['low'])} to "
                    f"{money_md(result['high'])}**. A range like this contained the actual "
                    "revenue for about 80% of the movies in our test."
                )
                for note in result["notes"]:
                    st.warning(note)

                # A simple real-data comparison that does not use the model at all.
                similar = df[df["budget"].between(budget * 0.5, budget * 2)]
                if len(similar) >= 10:
                    st.info(
                        f"For comparison: {len(similar):,} movies in the data had a budget "
                        f"between {money_md(budget * 0.5)} and {money_md(budget * 2)}. "
                        f"Their median revenue was {money_md(similar['revenue'].median())}."
                    )

        with st.expander("How reliable is this?"):
            metrics = bundle["test_metrics"]
            trained = bundle["trained_on"]
            st.markdown(
                f"- The model was trained on {trained['movies']:,} movies released "
                f"{trained['first_year']} to {trained['last_year']}.\n"
                f"- In a test on {metrics['test_movies']} later movies it had not seen "
                f"(2014 to 2017), a typical prediction was off by a factor of about "
                f"{metrics['typical_factor']:.1f}, and the average error was about "
                f"{money_md(metrics['mae_dollars'])}. The R-squared on the log scale was "
                f"{metrics['r2_log']:.2f}.\n"
                "- The range shown is only approximate. For future movies it may contain the "
                "real revenue less often than 80% of the time.\n"
                "- It uses only the budget, runtime, release month, genres, series membership "
                "and language. It cannot know about stars, marketing, reviews, competing "
                "releases or surprise hits.\n"
                "- Amounts are as recorded in the dataset (not adjusted for inflation), the data "
                "ends in 2017, and revenue is gross revenue, not profit."
            )
