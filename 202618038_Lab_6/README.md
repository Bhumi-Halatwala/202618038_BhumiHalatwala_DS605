# Feature Extraction and Machine Learning with Image and Text Data

**DS605: Fundamentals of Machine Learning — Lab Assignment 6**

| | |
|---|---|
| **Author** | Bhumi Halatwala |
| **Student ID** | 202618038 |
| **Course** | DS605 — Fundamentals of Machine Learning |
| **Image Dataset** | [Asphalt Crack Dataset - 400 Images](https://data.mendeley.com/datasets/xnzhj3x8v4/1) |
| **Text Dataset** | [Email Spam Classification Dataset - 5,172 Emails](kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/tasks) |

---

## Overview 
This assignment builds complete traditional-ML pipelines — from raw pixels and raw word counts to trained classifiers — without relying on CNNs, deep learning, or pretrained embeddings. Every preprocessing, feature-extraction, and modelling decision is driven by inspection of the data, not by default parameters.

Two tasks are addressed:

1. **Image task — crack detection.** Given 400 asphalt images (200 cracked, 200 clean), extract intensity and edge statistics with NumPy/OpenCV and train traditional classifiers.
2. **Text task — spam classification.** Given 5,172 pre-vectorized emails (3,000 word-count features, 71/29 ham/spam), train and compare classifiers across two representations.

A final Part C asks for one *justified* representation improvement, measured against the Part B baseline.

---

## Repository Layout

```
202618038_Lab_6.ipynb        # Full notebook: Part A, B, C with outputs and interpretations
data/
  asphalt/
    Cracks/                  # 200 cracked asphalt images (448×448×3)
    NonCracks/               # 200 clean asphalt images (448×448×3)
  emails.csv                 # 5172 emails × 3002 cols (ID + 3000 words + label)
README.md
```

---

## Pipeline Summary

```
Raw Image/Text → Preprocessing → Feature Extraction / Vectorization
              → Train-Test Split → ML Model → Evaluation → Representation Improvement
```

---

## Part A — Image Crack Classification

**Preprocessing.** All images verified as uniform 448×448×3, so no resize needed. Converted to grayscale — cracks are intensity anomalies, not colour anomalies.

**Feature extraction (9 features per image).**
- 7 intensity statistics justified by the pixel histogram: mean brightness, contrast (std), dark-pixel ratio (<100), bright-pixel ratio (>190), median, 10th percentile, 90th percentile.
- 2 edge features from Canny: edge count, edge density.

**Canny refinement.** Initial thresholds (50, 150) fired on asphalt texture — edge density ≈ 0.37 for *both* classes (i.e. no signal). Applied Gaussian blur (5×5) and raised thresholds to (100, 200): density fell to 0.140 (crack) vs 0.095 (non-crack), widening the class gap ~3.6×.

**Modelling.** Stratified 80/20 split, StandardScaler fit only on training data, three classifiers compared.

| Model | Accuracy | F1 | Train (s) | Predict (ms) |
|---|---|---|---|---|
| Logistic Regression | 0.875 | 0.875 | 0.01 | 0.2 |
| SVM (RBF) | 0.925 | 0.927 | 0.003 | 1.0 |
| **RandomForest** | **0.963** | **0.963** | 0.40 | 17.0 |

**Findings.**
- RandomForest wins on accuracy/F1 with only 3 errors out of 80, unbiased across classes.
- SVM-RBF biases toward crack recall (0.95), which is preferable for road inspection — missing a crack is costlier than a false alarm.
- LR's symmetric error profile (5+5) confirms the class boundary is **not purely linear**.
- RF is ~50–100× slower to predict than LR — negligible for 400 images, material at drone-survey scale.
- Top features by RF importance: `p90`, `mean_bright`, `bright_ratio`. Notably `p90` ranked #1 despite having the smallest mean class gap — a reminder that nonlinear models value split utility over average shift.

---

## Part B — Spam Classification

**Dataset.** 5,172 emails × 3,000 word-count columns, labels 1=spam / 0=ham. Class balance is **71/29** — accuracy alone is misleading, so precision/recall/F1 on the spam class are the primary metrics.

**Cleaning.** No all-zero columns; no empty emails. Only the ID column dropped.

**Two representations compared.**
- **Raw counts** — frequency of each word (values up to 2327).
- **Binary counts** — presence/absence (0/1), ignoring multiplicity.

**Modelling.** Same stratified 80/20 split for both representations; `class_weight="balanced"` on LR and LinearSVC to counteract the 71/29 skew.

**Raw counts:**

| Model | Acc | PrecSpam | RecSpam | F1Spam | Train (s) |
|---|---|---|---|---|---|
| MultinomialNB | 0.942 | 0.868 | 0.943 | 0.904 | 0.19 |
| **LogisticRegression** | **0.980** | **0.949** | **0.983** | **0.966** | 8.6 |
| LinearSVC | 0.969 | 0.944 | 0.950 | 0.947 | 7.4 |

**Binary counts:**

| Model | Acc | PrecSpam | RecSpam | F1Spam | Train (s) |
|---|---|---|---|---|---|
| MultinomialNB | 0.930 | 0.850 | 0.923 | 0.885 | 0.13 |
| LogisticRegression | 0.977 | 0.939 | 0.983 | 0.961 | 0.66 |
| LinearSVC | 0.974 | 0.942 | 0.970 | 0.956 | 1.20 |

**Findings.**
- **LR on raw counts is the best spam classifier** — F1Spam 0.966, only 5 missed spam and 16 false alarms.
- Word multiplicity carries real signal; binary throws it away at a ~0.5-point F1 cost.
- But binary counts train LR roughly an order of magnitude faster (~0.7 s vs ~8.6 s) — a strong operational trade-off driven by the numerical scale of raw counts.
- MultinomialNB consistently underperforms (F1 0.885–0.904) — its independence assumption is violated by correlated spam vocabulary.
- LinearSVC did *not* improve when `max_iter` was raised 3000 → 10000 — a reminder that more iterations ≠ better generalization.

---

## Part C — Vocabulary Pruning (Justified Improvement)

**Change.** Prune the vocabulary by document frequency: drop words appearing in **<5 emails** (rare, likely noise/typos) and words appearing in **>95% of emails** (near-universal, effectively stopwords).

**Result.** Only 30 words removed (3,000 → 2,970, −1%), yet the impact was disproportionate:

| Model | F1Spam (base) | F1Spam (pruned) | Train (base) | Train (pruned) |
|---|---|---|---|---|
| MultinomialNB | 0.904 | **0.924** | 0.19 | 0.13 |
| LogisticRegression | 0.966 | 0.961 | 8.6 | **3.2** |

**Findings.**
- LR training time fell **~60%** with a negligible 0.5-point F1Spam drop. Cause: the removed words carried the **largest integer counts** (up to 2327), dominating the optimizer's convergence cost. **Feature count is not the only cost driver — feature value scale is too.**
- MultinomialNB *improved* by +2 points of F1Spam (missed spam fell from 17 to 9). Removing stopwords cleaned up the independence assumption, exactly as theory predicts.
- **Net verdict:** the pruned representation is the better operational choice — LR catches 294 of 300 spam with only 18 false alarms while training in a third of the time; NB becomes competitive at negligible cost.

---

## A Note on Timing

All timings are wall-clock values on the same machine, reported as means over warm-up + repeated runs. Absolute values vary by 10–30% across runs and hardware, so all comparisons in this README are stated as **ratios** or rounded magnitudes, which are stable. The relative findings (e.g. "binary trains ~10× faster") reproduce reliably; the exact seconds do not.

---

## Key Takeaways

1. **Look at the data first.** The Canny refinement, the intensity thresholds, and the vocabulary pruning thresholds were all chosen *after* inspecting histograms.
2. **Feature scale matters as much as feature count.** Removing 30 high-magnitude words cut LR training by ~60% while removing 1% of columns.
3. **Match the metric to the task.** With 71/29 imbalance, accuracy lies; F1Spam and per-class recall tell the real story. For road inspection, crack recall matters more than false alarms.
4. **Nonlinear models earn their cost.** RF beats LR by 9 points on images — but is ~100× slower to predict.
5. **More iterations ≠ better generalization.** LinearSVC's accuracy *fell* when `max_iter` was raised, illustrating the bias-variance tug.
