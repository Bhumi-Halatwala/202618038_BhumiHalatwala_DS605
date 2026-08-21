# DS605: Fundamentals of Machine Learning

## Lab Assignment 3 — Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Bhumi Halatwala

**Student ID:** 202618038

**Course:** DS605 — Fundamentals of Machine Learning

---

## 1. Assignment Overview

This lab focuses on **data preprocessing, pipeline construction, classification, and model performance evaluation using Scikit-learn**.

The **Hotel Booking Demand** dataset from Kaggle is used to predict whether a hotel booking will be canceled. Two preprocessing approaches, **StandardScaler** and **MinMaxScaler**, are compared with two classification models: **Logistic Regression** and **Decision Tree Classifier**.

The main objective is to determine how preprocessing choices affect model performance while maintaining a consistent train-test split and model configuration.

---

## 2. Dataset

**Dataset:** Hotel Booking Demand
**Source:** Kaggle
**File:** `hotel_bookings.csv`

The dataset contains information related to hotel reservations, including booking details, customer information, stay duration, previous bookings, and cancellation status.

The target variable is **`is_canceled`**:

* `0` — Booking was not canceled
* `1` — Booking was canceled

**Dataset Link:** Kaggle Hotel Booking Demand

---

## 3. Data Preprocessing

The dataset was first explored using basic Pandas functions to understand its structure, dimensions, data types, descriptive statistics, and target class distribution.

The preprocessing included:

* Checking missing values and their percentages.
* Identifying columns with high missingness and deciding whether they should be removed.
* Removing **`reservation_status`** and **`reservation_status_date`** because they directly reveal the final booking outcome and can cause data leakage.
* Identifying potential outliers in selected numerical features using the IQR method and/or boxplots.
* Removing only clear and extreme outliers.
* Separating numerical and categorical features.

---

## 4. Preprocessing Pipelines

Two Scikit-learn preprocessing pipelines were created.

### Pipeline A — StandardScaler

Numerical features were processed using **KNN Imputation** followed by **StandardScaler**.

Categorical features were handled using **most-frequent imputation** followed by **One-Hot Encoding**.

### Pipeline B — MinMaxScaler

Numerical features were processed using **KNN Imputation** followed by **MinMaxScaler**.

The same categorical preprocessing was used as in Pipeline A.

`ColumnTransformer` and `Pipeline` were used to ensure that preprocessing was fitted only on the training data and applied consistently to the test data.

---

## 5. Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

Stratification was used to maintain a similar distribution of the target classes in both sets. The same split was used for all four experiments to ensure a fair comparison.

---

## 6. Classification Models

Two classification algorithms were evaluated:

* **Logistic Regression**
* **Decision Tree Classifier**

Each model was trained using both preprocessing pipelines, resulting in four experiments:

1. Logistic Regression + StandardScaler
2. Logistic Regression + MinMaxScaler
3. Decision Tree + StandardScaler
4. Decision Tree + MinMaxScaler

---

## 7. Model Evaluation

Each model was evaluated using:

* Training Accuracy
* Testing Accuracy
* Precision
* Recall
* F1-score

A final comparison table was created to compare the performance of all four model-pipeline combinations.

Confusion matrices were also generated for the best Logistic Regression and Decision Tree results.

The difference between training and testing performance was examined to identify possible **overfitting**.

---

## 8. Key Observations

The final results are discussed in the notebook based on the comparison table and confusion matrices.

The analysis focuses on:

* Identifying the best overall model and preprocessing combination.
* Comparing the effect of StandardScaler and MinMaxScaler on Logistic Regression.
* Determining whether scaling has a significant effect on Decision Tree performance.
* Comparing training and testing performance to identify possible overfitting.
* Using precision, recall, F1-score, and confusion matrices along with accuracy for a more complete evaluation.

---

## 9. Repository Contents

* **Jupyter Notebook** — Complete data preprocessing, pipeline creation, model training, evaluation, and observations.
* **Cleaned Dataset** — Dataset used for the final modeling process.
* **Confusion Matrix Figures** — Confusion matrices for the selected Logistic Regression and Decision Tree models.
* **README.md** — Assignment description, methodology, and final observations.

---

## 10. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 11. Conclusion

This assignment demonstrates an end-to-end machine learning workflow using Scikit-learn, from **data understanding and cleaning to preprocessing, model training, and performance evaluation**.

The comparison of StandardScaler and MinMaxScaler with Logistic Regression and Decision Tree provides an understanding of how preprocessing choices can affect different machine learning algorithms. The use of pipelines also ensures a structured and leakage-free preprocessing workflow.
