# Fundamentals of Machine Learning - Lab Assignment 2

## Student Information

**Name:** Bhumi Halatwala
**Student ID:** 202618038

## Assignment Title

**Lab 2 — NumPy and Pandas: Vectorized Programming and Titanic Data Wrangling**

## Dataset

**Kaggle Titanic Dataset (`train.csv`)**

The Titanic dataset contains passenger information such as passenger class, sex, age, family details, fare, port of embarkation, and survival status.

## Project Details

This assignment focuses on applying **NumPy** for vectorized numerical operations and **Pandas** for data wrangling and exploratory analysis.

### Part A — Vectorized Programming with NumPy

The following operations were implemented:

* Random array generation and statistical analysis
* Array creation using `arange()`, `zeros()`, `ones()`, and `linspace()`
* 2D and 3D arrays with indexing and slicing
* Reshaping and flattening arrays
* Vectorized arithmetic operations
* Matrix addition, element-wise multiplication, and matrix multiplication
* Matrix transpose, determinant, and inverse
* Normal distribution generation and histogram visualization

All numerical operations were performed using vectorized NumPy operations without explicit Python loops.

### Part B — Data Wrangling with Pandas

The Titanic dataset was analyzed using:

* Dataset inspection using `head()`, `tail()`, `shape`, `columns`, `info()`, and `describe()`
* Row and column selection using `loc` and `iloc`
* Boolean filtering and querying
* Grouping and aggregation using `groupby()`
* Missing-value analysis and imputation
* Fare outlier detection using the IQR method
* Feature engineering using `FamilySize` and `IsAlone`
* Pivot table analysis
* Correlation analysis and visualization
* Survival rate analysis by Sex and Passenger Class
* Age vs Fare visualization based on survival status

A cleaned version of the Titanic dataset and the generated visualizations are included in the repository.

## Key Observations

1. Female passengers had a considerably higher survival rate than male passengers.

2. First-class passengers generally had a higher survival rate than passengers in second and third class.

3. Third-class passengers had the lowest survival rate among the three passenger classes.

4. Passenger class and survival showed a negative relationship, indicating lower survival rates for higher class-number categories.

5. Passengers who survived generally had higher average fares than passengers who did not survive.

6. The Titanic dataset contains missing values, particularly in the `Age` column, which were handled using appropriate imputation techniques.

7. The `Fare` column contains high-value observations that were identified as outliers using the 1.5 × IQR rule.

## Repository Contents

```text
202618038_Lab_2/
│
├── README.md
├── 202618038_Lab02.ipynb
├── train.csv
├── titanic_cleaned.csv
│
└── Generated Figures
```

## Tools and Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook
