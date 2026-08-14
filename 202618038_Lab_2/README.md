# NumPy and Pandas Data Wrangling

**Name:** Bhumi Halatwala

**Student ID:** 202618038

## Project Overview

This project demonstrates fundamental data manipulation and analysis techniques using **NumPy** and **Pandas**. The assignment is divided into two parts.

**Part A** focuses on vectorized programming with NumPy, including array creation, statistical operations, indexing, slicing, reshaping, vectorized arithmetic, matrix operations, linear algebra, and generating data from a normal distribution.

**Part B** focuses on data wrangling and exploratory analysis using the Kaggle Titanic dataset. The analysis includes data inspection, filtering, grouping, aggregation, missing-value handling, outlier detection, feature engineering, pivot tables, and visualizations.

## Dataset

The project uses the **Kaggle Titanic `train.csv` dataset**.

The dataset contains information about Titanic passengers, including:

* Passenger class
* Sex
* Age
* Number of siblings/spouses
* Number of parents/children
* Fare
* Port of embarkation
* Survival status

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

## Assignment Tasks

### Part A — Vectorized Programming with NumPy

* Created and analyzed random numerical arrays.
* Performed statistical calculations including mean, median, minimum, maximum, and standard deviation.
* Used `zeros()`, `ones()`, `arange()`, and `linspace()`.
* Created and manipulated 2D and 3D arrays.
* Applied indexing, slicing, reshaping, and flattening.
* Performed vectorized arithmetic and matrix operations.
* Calculated matrix transpose, determinant, and inverse.
* Verified matrix inversion using `np.allclose()`.
* Generated normally distributed data and visualized it using a histogram.

### Part B — Titanic Data Wrangling with Pandas

* Loaded and inspected the Titanic dataset.
* Used `loc` and `iloc` for data selection.
* Applied Boolean filtering to answer specific passenger-related questions.
* Calculated survival rates using `groupby()` and aggregation.
* Analyzed missing values and applied different imputation methods.
* Detected Fare outliers using the IQR method.
* Created `FamilySize` and `IsAlone` features.
* Used pivot tables to compare survival rates across Sex and Pclass.
* Created correlation and survival visualizations.
* Analyzed the relationship between Age, Fare, and Survival.

## Key Observations

The analysis demonstrates several patterns in the Titanic dataset. Survival rates varied considerably by **Sex** and **Passenger Class**, with female and higher-class passengers generally showing higher survival rates. Fare and passenger class also showed meaningful relationships with survival. The dataset contained missing values, particularly in the Age-related data, and Fare included observations identified as outliers using the IQR method.

## Conclusion

This assignment provides practical experience with NumPy and Pandas for numerical computing and data wrangling. It demonstrates how vectorized operations, statistical analysis, data filtering, aggregation, feature engineering, missing-value treatment, outlier detection, and visualization can be applied to a real-world dataset.
