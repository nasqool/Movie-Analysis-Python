-- 01_financial_queries.sql
-- SQL analysis of movie revenues, budgets, genres and yearly trends.
-- Database: data/processed/movies.db  (tables: movies, movie_genres)
-- Only movies with a known budget AND revenue are used (has_financials = 1).
-- All money values are in US dollars as recorded in the dataset (not inflation-adjusted).

-- Q1: How many movies have known financials, and what are the typical values?
SELECT COUNT(*)                              AS movies,
       CAST(ROUND(AVG(budget)) AS INTEGER)   AS avg_budget,
       CAST(ROUND(AVG(revenue)) AS INTEGER)  AS avg_revenue,
       CAST(MAX(revenue) AS INTEGER)         AS highest_revenue
FROM movies
WHERE has_financials = 1;

-- Q2: Which 10 movies have the highest worldwide revenue?
SELECT title,
       release_year,
       CAST(budget AS INTEGER)  AS budget,
       CAST(revenue AS INTEGER) AS revenue
FROM movies
WHERE has_financials = 1
ORDER BY revenue DESC
LIMIT 10;

-- Q3: Which genres have the highest average revenue? (genres with at least 30 movies)
SELECT g.genre,
       COUNT(*)                                AS movies,
       CAST(ROUND(AVG(m.revenue)) AS INTEGER)  AS avg_revenue,
       CAST(ROUND(AVG(m.budget)) AS INTEGER)   AS avg_budget
FROM movies AS m
JOIN movie_genres AS g ON g.movie_id = m.id
WHERE m.has_financials = 1
GROUP BY g.genre
HAVING COUNT(*) >= 30
ORDER BY avg_revenue DESC;

-- Q4: How did recorded revenue change year by year since 2000?
SELECT release_year,
       COUNT(*)                                 AS movies,
       CAST(ROUND(AVG(revenue)) AS INTEGER)     AS avg_revenue,
       CAST(ROUND(SUM(revenue)) AS INTEGER)     AS total_revenue
FROM movies
WHERE has_financials = 1
  AND release_year >= 2000
GROUP BY release_year
ORDER BY release_year;

-- Q5: Which movies have the highest revenue multiple (revenue / budget)?
-- Only movies with a budget of at least 1 million dollars.
SELECT title,
       release_year,
       CAST(budget AS INTEGER)  AS budget,
       CAST(revenue AS INTEGER) AS revenue,
       ROUND(revenue / budget, 1) AS revenue_multiple
FROM movies
WHERE has_financials = 1
  AND budget >= 1000000
ORDER BY revenue_multiple DESC
LIMIT 10;

-- Q6: By decade, how many movies earned less than their budget?
-- (Revenue below budget does not automatically mean a financial loss.)
SELECT (release_year / 10) * 10 AS decade,
       COUNT(*) AS movies,
       SUM(CASE WHEN revenue < budget THEN 1 ELSE 0 END) AS below_budget,
       ROUND(100.0 * SUM(CASE WHEN revenue < budget THEN 1 ELSE 0 END) / COUNT(*), 1)
           AS percent_below_budget
FROM movies
WHERE has_financials = 1
GROUP BY decade
ORDER BY decade;