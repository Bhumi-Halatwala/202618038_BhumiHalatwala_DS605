# Airbnb Price Prediction — End-to-End ML Project

**DS605: Fundamentals of Machine Learning — Lab Assignment 4**

| | |
|---|---|
| **Author** | Bhumi Halatwala |
| **Student ID** | 202618038 |
| **Course** | DS605 — Fundamentals of Machine Learning |
| **Dataset** | [NYC Airbnb Open Data (AB_NYC_2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) |
| **Live Demo** | [airbnb-price-predictor.streamlit.app](https://202618038bhumihalatwalads605-qhctewgschfwbjoclqguka.streamlit.app/) |

---

## Overview

An end-to-end machine learning pipeline that predicts **nightly Airbnb prices in New York City** from listing attributes and serves the result through an interactive **Streamlit** app.

---

## What Was Done

**Data:** 48,895 listings × 16 columns. Dropped identifiers and unstructured text, removed invalid listings (price = 0, minimum_nights > 365, availability = 0), and capped price at the **99th percentile ($799)** to limit outlier influence. Final dataset: **30,984 × 11**.

**Missing values:** `last_review` and `reviews_per_month` were missing for the same 10,052 rows (listings never reviewed) — imputed `reviews_per_month = 0`.

**Feature engineering:**
- `reviews_per_month` → 0 for never-reviewed listings
- `host_type` — host listing count binned into Single / Small / Medium / Commercial

**Preprocessing (leak-free):** 80/20 train-test split performed **before** any transformation. Numeric → median imputation + scaling; Categorical → mode imputation + one-hot encoding, all wrapped in a `ColumnTransformer` pipeline fit only on training data.

**Models compared:** Linear Regression, Ridge, Random Forest, Gradient Boosting (5-fold CV). Gradient Boosting was tuned with `RandomizedSearchCV` (250 trees, depth 5, lr 0.1, subsample 0.8).

---

## Results

| Model | CV RMSE | CV R² |
|---|---|---|
| Linear Regression | 85.55 | 0.411 |
| Ridge | 85.50 | 0.411 |
| Random Forest | 77.39 | 0.517 |
| **Gradient Boosting (tuned)** | **77.63** | **0.491** |

**Final test-set performance:**

| Metric | Value |
|---|---|
| RMSE | **$77.28** |
| MAE | **$46.77** |
| R² | **0.514** |
| Train–Test R² gap | 0.105 (mild overfit, acceptable) |

**What drives price:** room type (`Entire home/apt`) dominates (~42% importance), followed by location (`longitude` + `latitude`, ~24%) and neighbourhood-level features. Review activity and host type contribute little.

---

## Repository Structure

```
202618038_Lab_4/
├── Visualizations/
│   ├── Bar Plots.png
│   ├── Corr Matrix.png
│   ├── Feature Importance.png
│   ├── Null Corr Map.png
│   ├── Pred VS Actual.png
│   ├── Residual.png
│   └── Target Distribution.png
├── 202618038_Lab04.ipynb
├── app.py
├── model_pipeline.pkl
├── requirements.txt
├── AB_NYC_2019.csv
└── README.md
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The saved `model_pipeline.pkl` bundles the preprocessor and trained model — no retraining needed.

---

## Limitations

- Underpredicts luxury listings (target capped at $799)
- 2019 data only — pre-COVID pricing
- No seasonality, amenities, images, or host response metrics
- R² ≈ 0.51 — realistic ceiling for 11 tabular features


## Author

**Bhumi Halatwala** — 202618038
DS605 · Fundamentals of Machine Learning
