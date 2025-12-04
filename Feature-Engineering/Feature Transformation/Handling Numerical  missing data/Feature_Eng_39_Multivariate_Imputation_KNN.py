#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Multivariate Imputation - KNN Imputer
#
# Definition:
# KNN Imputer handles missing values by leveraging the k-Nearest Neighbors algorithm.
# Instead of filling with global statistics (mean/median), it estimates missing values
# using the values from the k most similar samples.
#
# Why Important:
# - Preserves multivariate relationships between features.
# - More accurate than univariate imputation when features are correlated.
# - Useful in datasets where missingness is not random.

# ### Key Parameters
# #####  n_neighbors: Number of neighbors to use for imputation (default is 5).
# #####  weights: How to weight the neighbors ("uniform" or "distance").
# #####  metric: Distance metric to use (default is "nan_euclidean" in scikit-learn).
# #####  missing_values: Placeholder for missing values (default is np.nan)
# ============================================================

# ------------------------------------------------------------
# STEP 1: Import Libraries
# ------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv('train.csv')[['Age','Pclass','Fare','Survived']]
print("Initial dataset sample:\n", df.head())

# ------------------------------------------------------------
# STEP 3: Explore Missingness
# ------------------------------------------------------------
print("Percentage of missing values per column:\n", df.isnull().mean() * 100)

# INTERPRETATION: Age has missing values, while Pclass and Fare are mostly complete.
# KNN Imputer will use correlations between Age, Pclass, and Fare to estimate missing Age.

# ------------------------------------------------------------
# STEP 4: Prepare Features and Target
# ------------------------------------------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

print("Feature sample:\n", X.head())
print("Target sample:\n", y.head())

# ------------------------------------------------------------
# STEP 5: Train/Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print("Training sample:\n", X_train.head())

# ------------------------------------------------------------
# STEP 6: Prepare KNN Imputer
# ------------------------------------------------------------
# WHY: Use k=3 neighbors with distance weighting.
# WHAT: Closer neighbors contribute more to imputation.
# HOW: Euclidean distance computed over available features.
knn = KNNImputer(n_neighbors=3, weights='distance')

# ------------------------------------------------------------
# STEP 7: Fit & Transform
# ------------------------------------------------------------
X_train_trf = knn.fit_transform(X_train)
X_test_trf = knn.transform(X_test)

print("Transformed training sample:\n", X_train_trf[:5])
print("Transformed test sample:\n", X_test_trf[:5])

# INTERPRETATION: Missing Age values are replaced by weighted averages
# of Age from similar passengers (based on Pclass and Fare).

# ------------------------------------------------------------
# STEP 8: Train Logistic Regression
# ------------------------------------------------------------
# WHY: Logistic Regression is used to predict survival.
# WHAT: Imputed dataset ensures no missing values.
# HOW: Fit model on transformed training data.
# The fit_transform and transform methods of imputers (like KNNImputer and SimpleImputer) will return NumPy arrays, 
#  which is exactly what you want to feed into LogisticRegression.

lr = LogisticRegression()
lr.fit(X_train_trf, y_train)

y_pred = lr.predict(X_test_trf)
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# INTERPRETATION: Accuracy reflects how well imputation + model
# captured survival patterns. Compare with mean/median imputation
# to see improvement.

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates Multivariate Imputation using KNN Imputer.")
print("Key Insights:")
print("1. KNN Imputer leverages correlations between features to estimate missing values.")
print("2. Weighted distance ensures closer neighbors influence imputation more strongly.")
print("3. Logistic Regression trained on imputed data achieves stable accuracy.")
print("4. Compared to mean/median imputation, KNN preserves multivariate relationships better.")
print("\nRecruiter Takeaway: This script shows advanced preprocessing skills,")
print("awareness of multivariate impacts, and clear communication — essential traits for applied ML roles.")
print("============================================================")