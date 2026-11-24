# Movie Box Office Analysis & Success Prediction

A data analytics project that cleans a public movie dataset, explores which factors are associated with worldwide box office revenue, queries the data with SQL, and trains a machine learning model that estimates a movie's worldwide revenue from information known before its release. Everything is presented in an interactive Streamlit dashboard.

## Dataset

- **Source:** *The Movies Dataset* by Rounak Banik on Kaggle: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- **License:** CC0: Public Domain (as stated on the Kaggle page).
- **Origin:** The metadata was collected from the TMDB API and GroupLens. This project is not endorsed or certified by TMDb.
- **File used:** `movies_metadata.csv` (45,466 rows, 24 columns). It is not modified; all cleaning happens in code.
- **Coverage:** movies released on or before July 2017.
- **Caveats:** TMDB is crowd-edited, so budgets and revenues can be wrong or missing. The file does not state a currency (assumed US dollars). Amounts are not adjusted for inflation. Revenue is gross revenue, not profit, and gross revenue minus budget is not net profit.

## Data cleaning

| Step | Rows affected |
|---|---|
| Raw rows | 45,466 |
| Corrupted rows removed (values shifted into the wrong columns) | 3 |
| Identical duplicate rows removed | 17 |
| Extra rows with a repeated movie id removed | 13 |
| Movies not marked "Released" removed | 448 |
| Movies with no release date removed | 78 |
| **Cleaned released movies** | **44,907** |
| Of these, with a believable budget **and** revenue (both at least $1,000) | **5,303** |

A budget or revenue of 0 or below $1,000 is treated as "unknown", not as a real value. This is a judgement call. All analysis of money uses the 5,303 movies with both values known, which may not represent all movies.

## Key findings (from this dataset)

- The typical (median) movie among the 5,303 has a budget of $17.0M and revenue of $30.5M. The average revenue ($91.6M) is much higher, because a few hits pull it up. The highest recorded revenue is Avatar, about $2.79B.
- Animation has the highest average revenue among genres with at least 30 movies (about $225.7M); Documentary has the lowest (about $18.8M).
- Budget and revenue are positively related (Spearman correlation 0.70). This does not prove that a bigger budget causes a bigger revenue.
- About 29.5% of the movies have revenue below their budget. This is not the same as a financial loss.
- Median revenue is highest for movies released in June, July and December, and lowest for September. This may reflect which movies are scheduled in those months, not the months themselves.
- Audience ratings are only weakly related to revenue (Spearman 0.06 among 4,353 movies with at least 50 votes). Ratings and votes are collected after release, so they are not used for prediction.

## Machine learning

- **Target:** log10 of worldwide revenue (revenue spans thousands to billions of dollars).
- **Features, all known before release:** log budget, runtime, release month, part of a collection, English language, and 20 genre indicators.
- **Not used (to avoid data leakage):** popularity, vote average, vote count, and release year.
- **Validation:** chronological. Trained on 4,578 movies released up to 2013, tested on 725 movies released 2014 to 2017.

| Model | MAE ($M) | R² (log scale) | Typical error factor |
|---|---|---|---|
| Baseline (always the median) | 124.9 | -0.002 | 6.4 |
| Linear Regression, budget only | 95.2 | 0.448 | 3.6 |
| Linear Regression, all features | 75.9 | 0.511 | 3.3 |
| **Random Forest (final model)** | **69.3** | **0.517** | **3.3** |

The Random Forest was slightly better than Linear Regression in every test period we tried, but the edge is modest. The final model is retrained on all 5,303 movies.

**What this means:** a typical prediction is off by a factor of about 3, and the range shown in the dashboard is wide on purpose. The model cannot foresee surprise hits (for example low-budget horror hits) or flops.

## Limitations

- Predictions are rough estimates, not forecasts to rely on for financial decisions.
- Only budget, runtime, release month, genres, series membership and language are used. Stars, directors, marketing, competition and reviews are not.
- Only 5,303 of 44,907 movies have usable budget and revenue figures.
- Data ends in 2017. Money is not adjusted for inflation.
- Budgets and revenues in the source may contain errors.
- The models were not tuned, and one main test split plus a few cut-off years were used.

## How to run (Windows, PowerShell)

1. Create and activate a virtual environment, and install the libraries:

```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. Download `movies_metadata.csv` from the Kaggle page and place it in `data/raw/`.
3. Run the pipeline from the project folder:

```
python src\clean_data.py
python src\build_database.py
python src\run_sql.py sql\01_financial_queries.sql
python src\eda_01_overview.py
python src\eda_02_genres_budget.py
python src\eda_03_roi_timing_ratings.py
python src\train_models.py
python src\evaluate_models.py
python src\save_model.py
```

4. Start the dashboard:

```
python -m streamlit run app.py
```

## Project structure

```
Movie_Box_Office_Analysis/
├── app.py                 Streamlit dashboard
├── requirements.txt       library versions
├── data/
│   ├── raw/               original dataset (never edited)
│   └── processed/         cleaned CSV files and the SQLite database
├── sql/                   SQL queries
├── src/                   Python scripts (cleaning, EDA, SQL, features, models, prediction)
├── models/                saved model (Joblib)
└── outputs/               charts and model results
```

## Tools

Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, SQLite (SQL), Scikit-learn, Streamlit, Altair, Joblib.

## Author

[Salman Siraj]