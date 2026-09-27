# Machine Learning with Scikit-learn and From Scratch

**DS605: Fundamentals of Machine Learning — Lab Assignment 5**

| | |
|---|---|
| **Author** | Bhumi Halatwala |
| **Student ID** | 202618038 |
| **Course** | DS605 — Fundamentals of Machine Learning |
| **Dataset** | [UCI Productivity Prediction of Garment Employees](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees) |

---

## Repository Contents

| File | Description |
|---|---|
| `202618038_Lab05.ipynb` | Complete notebook — preprocessing, Part A (sklearn), Part B (from scratch), Part C (optimization) |
| `garments_worker_productivity.csv` | Raw dataset |
| `README.md` | This file |

## How to Run

Open the notebook and run top-to-bottom. Requires only `numpy`, `pandas`, and `scikit-learn`. Preprocessed arrays and the frozen train/test split stay in memory and are reused across all three parts.

## Task Definition

- **Regression target:** `actual_productivity`
- **Classification target:** `MeetsTarget = 1 if actual_productivity >= targeted_productivity, else 0`
- **Class balance:** ~73% / 27% (imbalanced)

---

## Preprocessing Decisions

Every step was driven by what the data showed, not by a default template.

| Decision | Justification |
|---|---|
| Median imputation for `wip` | 42% missing and right-skewed (mean 1190 > median 1039, max 23122) — mean would distort the distribution |
| Added `wip_missing` flag | Missingness itself may carry signal; the model decides |
| Dropped `date` | 59 unique values, redundant with `quarter`+`day`, no ordinal meaning |
| Stripped whitespace, fixed `sweing` typo | `'finishing '` was counted as a separate category from `'finishing'` |
| One-hot with `drop_first=True` | Low-cardinality categoricals; avoids the dummy variable trap |
| Scaled continuous features only | One-hot columns are already 0/1 — scaling hurts interpretability |
| All statistics fit on train only | Median, mean, std computed on train to prevent leakage |

---

## Results

### Regression — LinearRegression

| Implementation | MAE | RMSE | R² | Train | Predict |
|---|---|---|---|---|---|
| sklearn | 0.1019 | 0.1418 | 0.3475 | 5.33 ms | 2.78 ms |
| from scratch | 0.1019 | 0.1418 | 0.3475 | **2.88 ms** | **0.13 ms** |

Manual implementation matches sklearn to **machine precision** (max prediction diff `1.83e-15`) and runs **~1.9× faster** — both use SVD internally, but sklearn adds input-validation overhead on every call.

### Classification — LogisticRegression

| Model | Accuracy | F1 (weighted) | Recall C0 | Recall C1 | Train |
|---|---|---|---|---|---|
| sklearn (lbfgs) | 0.746 | 0.726 | 0.377 | 0.895 | **24 ms** |
| from scratch — full-batch GD | 0.750 | 0.720 | 0.319 | 0.924 | 170 ms |
| from scratch — SGD + momentum | **0.762** | **0.738** | 0.362 | 0.924 | 96 ms |
| from scratch — SGD + class weight | 0.704 | 0.718 | **0.841** | 0.649 | 62 ms |

---

## Key Observations

**1. Closed-form problems are trivial to match.**  
Linear regression via `np.linalg.lstsq` produces bit-identical predictions to sklearn's `LinearRegression` — same SVD solver. The manual version is faster because there's no wrapper overhead.

**2. Optimizer choice matters more than the code.**  
Full-batch gradient descent ran all 2000 iterations without converging. Mini-batch SGD with momentum (`lr=0.2, β=0.9, batch=64`) converged quickly and **beat sklearn on both accuracy (0.762 vs 0.746) and weighted F1 (0.738 vs 0.726)**.

**3. Class imbalance is orthogonal to optimizer choice.**  
Both sklearn and the unweighted SGD scored ~36% recall on the minority class. No optimizer fixed this. Only inverse-frequency class weights did — lifting class 0 recall from 0.36 to 0.84, at the cost of ~2 points of weighted F1 and ~28 points of class 1 recall.

**4. The runtime gap with sklearn is expected and explainable.**  
sklearn's lbfgs is compiled and uses second-order curvature information. The manual SGD is a first-order Python loop. Beating sklearn on raw speed was never the goal — matching or exceeding it on quality was, and it was achieved.

---

## Summary

- Hand-written implementations can match sklearn exactly on closed-form problems and beat it on iterative ones when the optimizer is right.
- On imbalanced data, per-class recall and F1 are the informative metrics — accuracy hides the failure mode.
- The engineering effort of writing and tuning SGD is real, but so is the payoff in transparency: class weighting was a one-line change in the gradient.
