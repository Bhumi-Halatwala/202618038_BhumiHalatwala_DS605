# Decision Trees and Support Vector Machines

**DS605: Fundamentals of Machine Learning — Lab Assignment 8**

| | |
|---|---|
| **Author** | Bhumi Halatwala |
| **Student ID** | 202618038 |
| **Course** | DS605 — Fundamentals of Machine Learning |
| **Dataset** | [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) |

---

## Overview

This lab explores two important supervised learning methods: **Decision Trees** and **Support Vector Machines (SVMs)**. It covers their underlying principles, implementation, evaluation, and comparison using the Banknote Authentication dataset, along with experiments on synthetic data to understand nonlinear classification.

## Objectives

- Understand impurity measures used in decision trees, including Gini impurity and entropy.
- Implement decision-tree splitting and classification from scratch.
- Compare a manually implemented decision tree with Scikit-learn's implementation.
- Study the effect of tree depth and splitting criteria on classification performance.
- Implement a linear SVM using hinge loss and regularization.
- Examine the effects of hyperparameters on SVM performance.
- Explore linear and nonlinear decision boundaries using different SVM kernels.
- Evaluate and compare models using appropriate classification metrics.

## Dataset

The **Banknote Authentication** dataset is provided by the UCI Machine Learning Repository. It contains numerical features extracted from images of genuine and forged banknote specimens.

The dataset contains **1,372 instances and 5 columns**: four input features and one target variable.

| Feature | Description |
|---|---|
| `variance` | Variance of the wavelet-transformed image |
| `skewness` | Skewness of the wavelet-transformed image |
| `curtosis` | Curtosis of the wavelet-transformed image |
| `entropy` | Entropy of the image |
| `class` | Target label indicating genuine or forged banknotes |

The dataset has no missing values. The notebook also examines duplicate records and the distribution of the target classes before model training.

## Methodology

### Part A: Decision Trees

- Load and explore the dataset.
- Select two features for initial visualization and experimentation.
- Calculate Gini impurity and entropy.
- Determine the best split using impurity reduction.
- Implement a decision tree classifier from scratch.
- Compare its predictions and performance with Scikit-learn's decision tree.
- Investigate the effect of tree depth and splitting criteria.
- Extend the manual decision tree to use all four input features.

### Part B: Support Vector Machines

- Implement a linear SVM from scratch using hinge loss and regularization.
- Evaluate the classifier using accuracy, precision, recall, and F1-score.
- Investigate the effect of SVM hyperparameters.
- Use synthetic `make_moons` data to examine nonlinear classification.
- Compare linear and RBF-kernel SVMs.

### Model Comparison

The notebook compares the manual implementations and Scikit-learn models using a common train-test split where applicable. It presents the results in tables to facilitate comparison of their classification performance.

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Results

The notebook evaluates the models on the held-out test set and reports their classification performance.

In the saved notebook outputs:

- The manual decision tree using two features achieved approximately **93.09% accuracy**, matching the Scikit-learn decision tree's predictions.
- The manual decision tree using all four features achieved approximately **98.55% accuracy**.
- The RBF-kernel SVM achieved **100% accuracy** on the displayed test split.

These results apply to the recorded train-test split and should not be interpreted as a guarantee of performance on unseen datasets.

## Repository Contents

| File | Description |
|---|---|
| `202618038_Lab08.ipynb` | Notebook containing the implementations, experiments, visualizations, and model evaluations |
| `data_banknote_authentication.txt` | Banknote Authentication dataset required to execute the notebook |
| `VisualizationPlots` | Plots crteated throughout this assignment |

## Conclusion

This lab demonstrates how decision trees classify data through feature-based splits and how SVMs construct decision boundaries using margin maximization. The experiments compare manual implementations with established Scikit-learn models and illustrate how feature selection, tree complexity, regularization, and kernel choice affect classification performance.