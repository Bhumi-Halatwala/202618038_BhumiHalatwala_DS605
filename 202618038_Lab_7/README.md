# Image Representation, PCA, Eigenfaces and Face Recognition

**DS605: Fundamentals of Machine Learning — Lab Assignment 7**

| | |
|---|---|
| **Author** | Bhumi Halatwala |
| **Student ID** | 202618038 |
| **Course** | DS605 — Fundamentals of Machine Learning |
| **Dataset** | [AT&T Database of Faces (ORL)](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) |

---

## 1. Overview

This project explores image representation, dimensionality reduction, eigenfaces, and face recognition using the AT&T Database of Faces, commonly known as the ORL face dataset.

Each image is represented initially as a high-dimensional pixel vector. Principal Component Analysis (PCA) is then implemented using NumPy linear algebra to extract the principal directions of variation and represent faces in a lower-dimensional feature space. The resulting representations are evaluated through image reconstruction, storage comparisons, nearest-neighbor matching, and traditional machine-learning classifiers.

The experiment emphasizes the relationship between dimensionality reduction, information preservation, computational representation, and recognition performance.

## 2. Objectives

The main objectives of this assignment are to:

- Represent grayscale face images as matrices and flattened feature vectors.
- Construct a feature matrix and associate each image with its subject label.
- Split the dataset into training and testing sets while preserving subject representation.
- Calculate the mean face and center the training data.
- Implement PCA using Gram-matrix eigendecomposition and Singular Value Decomposition (SVD).
- Compare the principal directions obtained through both approaches.
- Determine the minimum number of components required to retain at least 90% of the variance.
- Visualize eigenfaces and reconstruct images using reduced-dimensional representations.
- Compare the storage requirements of original pixel vectors and PCA coefficients.
- Evaluate raw-pixel and PCA-space nearest-neighbor face recognition.
- Compare PCA-based representations using K-Nearest Neighbors, Logistic Regression, and Support Vector Machine classifiers.
- Evaluate recognition performance using accuracy, precision, recall, F1-score, and misclassification analysis.

## 3. Dataset

The project uses the [AT&T Database of Faces (ORL)](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces), which contains grayscale face images organized by subject.

| Property | Value |
|---|---:|
| Total images | 400 |
| Number of subjects | 40 |
| Images per subject | 10 |
| Image dimensions | 112 × 92 pixels |
| Features per flattened image | 10,304 |
| Pixel intensity range observed | 0–251 |

Each image is converted to grayscale and flattened into a one-dimensional vector containing 10,304 pixel intensities. The resulting feature matrix has dimensions **400 × 10,304**.

### Dataset organization

Place the extracted dataset in the following directory structure relative to the notebook:

```text
202618038_Lab_7/
├── README.md
├── 202618038_Lab7.ipynb
└── Dataset/
    ├── s1/
    │   ├── 1.pgm
    │   ├── 2.pgm
    │   └── ...
    ├── s2/
    └── ...
```

The dataset loader searches for subject folders containing `.pgm` images. If your extracted dataset uses a different folder name, update the `DATA_DIR` variable in the notebook accordingly.

The dataset is not included in this repository; download it from the linked source and place it in the expected directory before running the notebook.

## 4. Methodology

### 4.1 Image representation and preprocessing

Each image is loaded as a grayscale image and represented as a two-dimensional pixel matrix. The matrix is flattened to form a feature vector, and the subject directory supplies the corresponding class label.

A reproducible, stratified 70:30 train-test split is used:

| Partition | Images | Images per subject |
|---|---:|---:|
| Training set | 280 | 7 |
| Testing set | 120 | 3 |

All 40 subjects are represented in both partitions. The training set is used to estimate the PCA representation, while the held-out test set is reserved for final recognition evaluation.

### 4.2 Mean face and data centering

The mean face is calculated from the training images and subtracted from each training vector. This centers the training data before PCA is performed.

The mean face provides a reference representation, while centering allows PCA to identify directions of variation around that mean.

### 4.3 PCA using eigendecomposition and SVD

PCA is implemented using NumPy linear algebra rather than `sklearn.decomposition.PCA`.

Two approaches are examined:

1. **Gram-matrix eigendecomposition:** Eigenvectors of the smaller training-sample Gram matrix are computed and mapped back into the original pixel feature space.
2. **Singular Value Decomposition:** SVD is applied to the centered training matrix to obtain principal directions and singular values.

The leading principal directions from both methods are compared using absolute cosine similarity.

### 4.4 Explained variance and eigenfaces

The explained variance associated with the principal components is used to calculate cumulative explained variance. The minimum number of components needed to retain at least 90% of the training-data variance is then identified.

The principal directions are visualized as eigenfaces, which represent patterns of variation in facial appearance. Test images are projected into the reduced-dimensional space and reconstructed to examine the information preserved by PCA.

### 4.5 Storage comparison

The storage requirements of the original flattened image matrix and the PCA coefficient matrix are compared using their NumPy array sizes.

This comparison concerns the image representations only. It excludes the additional storage required for the mean face and principal component matrix used for encoding and reconstruction.

### 4.6 Nearest-neighbor recognition

A nearest-neighbor baseline is evaluated in two feature spaces:

- **Raw-pixel space:** Euclidean distances are calculated between flattened test and training images.
- **PCA space:** Euclidean distances are calculated between the corresponding PCA representations.

For each test image, the label of the nearest training image is used as the predicted subject. The two approaches are compared using held-out test accuracy and correct and incorrect match counts.

### 4.7 Traditional machine-learning classifiers

Three classifiers are evaluated using PCA-transformed features:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM) with an RBF kernel

Hyperparameter tuning is performed using cross-validation and `GridSearchCV`. PCA is implemented as a custom NumPy-based transformer inside a scikit-learn pipeline, ensuring that PCA is fitted within each cross-validation training fold.

The classifiers are compared using cross-validation accuracy, held-out test accuracy, and the selected hyperparameters. The final SVM model is also evaluated using a classification report and misclassification analysis.

## 5. Experimental Results

### 5.1 PCA and dimensionality reduction

| Metric | Result |
|---|---:|
| Original features per image | 10,304 |
| Minimum components for at least 90% variance | 91 |
| Cumulative variance at 91 components | 90.0473% |
| Cumulative variance at 90 components | 89.9268% |
| Maximum deviation from perfect similarity between leading PCA directions | 2.55 × 10⁻¹⁵ |

The eigendecomposition and SVD results show near-identical leading principal directions. A total of 91 components is required to cross the 90% cumulative explained-variance threshold.

### 5.2 Reconstruction and storage

| Metric | Result |
|---|---:|
| Number of PCA components | 91 |
| Reconstruction RMSE for five test images | 19.7143 |
| Original pixel matrix storage | 32,972,800 bytes |
| PCA coefficient matrix storage | 291,200 bytes |
| Reduction in coefficient-array storage | 99.12% |

PCA produces a substantially smaller image representation while retaining more than 90% of the variance. Reconstruction is not lossless because the discarded components contain information that is not represented in the reduced feature space.

### 5.3 Nearest-neighbor recognition

| Method | Correct predictions | Test accuracy |
|---|---:|---:|
| Raw-pixel nearest neighbor | 116/120 | 96.67% |
| PCA nearest neighbor (91 components) | 116/120 | 96.67% |

Both nearest-neighbor approaches achieve the same test accuracy on the selected split. The PCA representation therefore reduces the feature dimensionality substantially without reducing the observed recognition accuracy in this experiment.

### 5.4 Traditional classifier comparison

| Classifier | Selected PCA components | Best CV accuracy | Test accuracy |
|---|---:|---:|---:|
| Logistic Regression | 30 | 93.93% | 96.67% |
| SVM (RBF) | 30 | 93.93% | 96.67% |
| KNN | 30 | 92.14% | 93.33% |

Logistic Regression and RBF SVM achieved the highest held-out test accuracy, correctly predicting 116 of 120 test images. KNN correctly predicted 112 of 120 test images.

All three classifiers selected 30 components during hyperparameter tuning. This illustrates that retaining the maximum possible variance and selecting features for classification performance are different objectives.

### 5.5 Final SVM evaluation

| Metric | Result |
|---|---:|
| Test accuracy | 96.67% |
| Correct predictions | 116 |
| Incorrect predictions | 4 |
| Macro-average F1-score | Approximately 0.97 |
| Weighted-average F1-score | Approximately 0.97 |

The final SVM classification report shows strong performance across the 40 subjects. Some subjects have recall below 1.00 because one of their three test images was misclassified. The classification report and error visualizations provide additional detail about these cases.

*All reported values correspond to the current notebook execution, dataset split, and selected hyperparameters. Results may differ if the dataset, random seed, software versions, or experimental settings change.*

## 6. Key Observations

1. **Dimensionality reduction:** PCA reduces the feature count from 10,304 to 91 components while retaining 90.0473% of the training-data variance.
2. **Numerical agreement:** The leading principal directions obtained from Gram-matrix eigendecomposition and SVD agree to near machine precision.
3. **Compact representation:** The PCA coefficient matrix requires approximately 99.12% less storage than the original pixel matrix, excluding the PCA model overhead.
4. **Recognition performance:** Raw-pixel and 91-component PCA nearest-neighbor matching both achieve 96.67% test accuracy.
5. **Classifier comparison:** Logistic Regression and RBF SVM outperform KNN on this held-out test set.
6. **Variance versus classification:** The classifiers select 30 components even though 91 are required to retain 90% of the variance. Explained variance alone does not determine classification accuracy.
7. **Evaluation limitations:** The held-out test set contains only three images per subject, so individual subject-level metrics can change noticeably with a different split.

## 7. Technologies Used

- **Python** — implementation and experimentation
- **NumPy** — matrix operations, eigendecomposition, SVD, PCA projections, and reconstruction
- **Pandas** — organizing experimental results and generating comparison tables
- **Matplotlib** — image visualization, eigenfaces, reconstruction comparisons, and result plots
- **Pillow (PIL)** — loading and processing grayscale images
- **scikit-learn** — train-test splitting, pipelines, cross-validation, hyperparameter tuning, classification, and evaluation metrics

The PCA computation is implemented using NumPy linear algebra. Scikit-learn is used for the traditional classification models and evaluation workflow, not for its built-in PCA estimator.

## 8. Conclusion

This assignment demonstrates the application of PCA and eigenfaces to image representation and face recognition. NumPy-based eigendecomposition and SVD provide consistent principal directions, while PCA substantially reduces the dimensionality and storage requirements of the image representations.

On the selected train-test split, PCA-based nearest-neighbor recognition preserves the accuracy of raw-pixel matching, and Logistic Regression and RBF SVM achieve 96.67% test accuracy. The results also demonstrate that the number of components required to retain a chosen percentage of variance may differ from the number that performs best for classification.

Overall, the experiment highlights the usefulness of PCA as a dimensionality-reduction technique for face recognition while illustrating the importance of reconstruction quality, validation, and held-out evaluation.
