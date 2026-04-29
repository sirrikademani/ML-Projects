# 🎬 MovieLens Rating Prediction Pipeline

An end-to-end Data Science pipeline built on the **MovieLens 32M dataset** — covering data cleaning, exploratory analysis, statistical hypothesis testing, and machine learning models for both regression and classification.

---

## 📌 Problem Statement

Streaming platforms rely on recommendation engines to drive engagement. This project tackles two core prediction tasks:

- **Regression** — predict the exact rating (0.5–5.0) a user gives a movie
- **Classification** — predict whether a user will like a movie (rating > 3.0)

The goal is to build a clean, well-validated baseline pipeline that mirrors real-world DS workflows — from raw messy data to model-ready insights.

---

## 📂 Dataset

**MovieLens 32M** — 32 million ratings from real users, collected by GroupLens Research at the University of Minnesota.

| File | Description |
|---|---|
| `ratings.csv` | userId, movieId, rating, timestamp |
| `movies.csv` | movieId, title, genres |
| `tags.csv` | userId, movieId, tag, timestamp |
| `links.csv` | movieId, imdbId, tmdbId |

**Download:** https://grouplens.org/datasets/movielens/

---

## 🗂️ Project Structure

```
movielens-rating-prediction/
│
├── movielens_rating_prediction_pipeline.py   # main pipeline
├── README.md                                  # this file
├── data/
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv
│   └── links.csv
└── outputs/
    └── model_results.txt
```

---

## ⚙️ Pipeline Overview

```
Raw Data (32M rows)
      ↓
Phase 1 — Load & Clean
      ↓
Phase 2 — EDA & Feature Engineering
      ↓
Phase 3 — Statistical Hypothesis Testing
      ↓
Phase 4 — Modelling (Regression + Classification)
      ↓
Results & Insights
```

---

## 🔧 Phase 1 — Data Cleaning

- Loaded all 4 files with correct separators and encodings
- Converted Unix timestamps to datetime, extracted `rating_year` and `rating_month`
- Extracted `movie_age` and release `year` from title strings using regex
- Exploded pipe-separated genres into one-hot encoded dummy columns
- Removed duplicate genre columns from erroneous concat
- Applied **core filtering** — kept movies with ≥10 ratings and users with ≥5 ratings
- Took a stratified 10% sample for computational efficiency

| Stage | Rows | Movies | Users |
|---|---|---|---|
| Raw (10% sample) | 3,200,020 | 42,809 | 197,270 |
| After core filter | 2,964,172 | 12,526 | 134,940 |
| After dropna | 2,961,175 | — | — |

---

## 📊 Phase 2 — EDA & Feature Engineering

### Key findings

- **Rating distribution** is left-skewed — most ratings cluster at 3.0–4.0, indicating positive selection bias (users rate movies they chose to watch)
- **Power law** on both users and movies — 50% of users have ≤8 ratings, 50% of movies have ≤3 ratings
- **63.4% of ratings are positive** (liked) — class imbalance noted for classification

### Engineered features

| Feature | Description |
|---|---|
| `avg_rating_per_movie` | Mean rating for the movie across all users |
| `rating_count_per_movie` | Number of ratings received by the movie |
| `avg_rating_per_user` | Mean rating given by the user (user bias) |
| `rating_count_per_user` | Number of ratings given by the user |
| `log_movie_count` | Log-transformed movie rating count (handles power law) |
| `log_user_count` | Log-transformed user rating count (handles power law) |
| `movie_age` | Years between movie release and rating date |
| `liked` | Binary target — 1 if rating > 3.0, else 0 |
| Genre dummies | 19 one-hot encoded genre columns |

---

## 🧪 Phase 3 — Hypothesis Testing

All tests used **Welch's two-sample t-test** (`equal_var=False`) at α = 0.05.

> ⚠️ Note: With n = 2.96M, all tests will achieve statistical significance. **Effect size (mean difference) is the more honest metric at this scale.**

| Test | Group A | Group B | Mean Diff | t-stat | p-value | Practical? |
|---|---|---|---|---|---|---|
| Movie age | Old (<1990): 3.697 | New (≥1990): 3.492 | +0.205 | 146.26 | <0.001 | ✅ Yes |
| User activity | Prolific (>100): 3.250 | Casual (≤10): 3.765 | -0.515 | -235.48 | <0.001 | ✅ Yes |
| Genre | Action: 3.468 | Romance: 3.540 | -0.072 | -39.24 | <0.001 | ❌ No |
| Movie quality | High (≥4.0): 4.113 | Low (<3.0): 2.671 | +1.442 | 633.90 | <0.001 | ✅ Yes |

### Key insights

- **Old movies rated higher** — survivorship bias (only classics survive) and nostalgia effect
- **Prolific users rate harsher** — casual users only rate movies they loved (selection bias upward)
- **Genre is a weak signal** — 0.07 star difference between Action and Romance is statistically significant but practically irrelevant
- **Movie quality dominates** — 1.44 star difference between high and low quality movies is the strongest effect

> These are **observational findings**, not causal — groups were not randomly assigned, so confounders exist. This is not an A/B test.

---

## 🤖 Phase 4 — Modelling

### Feature matrix (26 features)

- User features: `avg_rating_per_user`, `rating_count_per_user`, `log_user_count`
- Movie features: `avg_rating_per_movie`, `rating_count_per_movie`, `log_movie_count`, `movie_age`
- Genre features: 19 one-hot encoded genre columns

### Train / Test Split

```
Train: 2,368,940 rows (80%)
Test:  592,235 rows  (20%)
Stratified on liked to preserve 63.3/36.7 class balance
```

---

### Regression — Linear Regression

Predict exact rating (0.5–5.0)

| Metric | Score |
|---|---|
| RMSE | 0.8394 |
| R² | 0.3621 |
| Baseline RMSE (predict mean) | 1.0510 |
| **Improvement over baseline** | **20%** |

**Top features by coefficient:**

| Feature | Coefficient | Interpretation |
|---|---|---|
| `avg_rating_per_user` | +0.902 | Generous raters rate everything higher |
| `avg_rating_per_movie` | +0.859 | Quality movies attract quality ratings |
| `log_user_count` | +0.044 | More active users rate slightly higher |
| `IMAX` | -0.049 | IMAX movies rated slightly lower |
| `log_movie_count` | -0.015 | More popular movies rated slightly lower |

---

### Classification — Logistic Regression

Predict whether user liked the movie (rating > 3.0)

| Metric | Score |
|---|---|
| Accuracy | 75.0% |
| Baseline accuracy (always predict liked) | 63.3% |
| AUC-ROC | **0.8057** |
| F1 (liked) | 0.81 |
| F1 (not liked) | 0.62 |

**Confusion Matrix:**

```
                 Predicted
                 Not liked    Liked
Actual Not liked  119,480     97,590
       Liked        50,350    324,815
```

- **Liked recall = 87%** — catches 87% of movies the user will enjoy ✅
- **Not liked recall = 55%** — misses 45% of movies user won't enjoy ⚠️
- Model is biased toward predicting liked due to class imbalance

---

## 📈 Results Summary

| Model | Key Metric | Score | vs Baseline |
|---|---|---|---|
| Linear Regression | RMSE | 0.8394 | +20% improvement |
| Linear Regression | R² | 0.3621 | — |
| Logistic Regression | AUC-ROC | 0.8057 | — |
| Logistic Regression | Accuracy | 75.0% | +11.7% improvement |

---

## ⚠️ Limitations

- Linear models cannot capture non-linear user-movie interactions
- No collaborative filtering — model doesn't know "users like Alice tend to agree with users like Bob"
- Class imbalance causes over-prediction of liked class
- Genre features proved weak predictors — richer content features needed

---

## 🚀 Next Steps

- [ ] Gradient Boosting (XGBoost / LightGBM) for non-linear relationships
- [ ] Handle class imbalance with `class_weight='balanced'` or SMOTE
- [ ] Matrix Factorisation (SVD / ALS) for collaborative filtering
- [ ] NLP features from tags using TF-IDF or sentence embeddings
- [ ] Hyperparameter tuning with cross-validation
- [ ] Deploy as a REST API with FastAPI

---

## 🛠️ Requirements

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
jupyter
```

Install with:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
```

---

## 👤 Author

Built as an end-to-end DS portfolio project demonstrating real-world data cleaning, EDA, statistical testing, and ML modelling on a large-scale dataset.

---

## 📄 License

This project uses the MovieLens dataset which is available for non-commercial use. See [GroupLens Terms](https://grouplens.org/datasets/movielens/) for details.
