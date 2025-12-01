#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Executive Summary: Hyperparameter Tuning with GridSearchCV
# ============================================================

# Purpose:
# This script demonstrates hyperparameter tuning (optimization) using
# scikit-learn’s GridSearchCV. Hyperparameter tuning is the process of
# selecting the best configuration settings for a model before training.

# Why it matters:
# Hyperparameters directly influence model performance. By systematically
# searching through candidate values, we can identify the optimal settings
# that maximize accuracy and generalization.

# Dataset:
# Iris dataset — contains measurements of iris flowers (features) and species labels (targets).

# Workflow:
# 1. Load and preprocess the dataset
# 2. Train/test split
# 3. Define models (RandomForestClassifier, SVC)
# 4. Specify hyperparameter search space
# 5. Apply GridSearchCV for tuning
# 6. Report best parameters and scores
#
# ============================================================

# ==== Step 1: Import Libraries ====

# NumPy and Pandas for data handling, Matplotlib/Seaborn for visualization.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # clean, professional plot style

# ==== Step 2: Load Dataset ====

# The Iris dataset is a classic dataset for classification.
from sklearn.datasets import load_iris
Iris = load_iris()
print("Iris dataset loaded successfully.")

# Inspect dataset object
Iris  # contains .data (features) and .target (labels)

# ==== Step 3: Transform Dataset ====

# Features (X) = measurements of iris flowers
# Target (Y) = species labels
X = Iris.data
print("Feature matrix:\n", X)

Y = Iris.target
print("Target vector:\n", Y)

# ==== Step 4: Train/Test Split ====

# train_test_split is a utility from scikit-learn that splits data into
# training and testing sets. test_size=0.2 means 20% of data is reserved for testing.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# ==== Step 5: Import RandomForestClassifier ====

# Random Forest is an ensemble method that builds multiple decision trees
# and averages their predictions for better accuracy and robustness.
from sklearn.ensemble import RandomForestClassifier

# ==== Step 6: Import GridSearchCV ====

# GridSearchCV systematically searches through hyperparameter values
# using cross-validation to find the best configuration.
from sklearn.model_selection import GridSearchCV

# ==== Step 7: Define Hyperparameter Search Space ====

# Hyperparameters:
# - n_estimators: number of trees in the forest
# - max_depth: maximum depth of each tree
# We use Python's range() to generate candidate values.
hyperparameter = {'n_estimators': list(range(1, 10)),
                  'max_depth': list(range(1, 5))}
print("Hyperparameter search space defined:", hyperparameter)

# ==== Step 8: Apply GridSearchCV (Random Forest) ====

model = GridSearchCV(RandomForestClassifier(), hyperparameter)
model.fit(X_train, y_train)
print("GridSearchCV fitting completed.")

# ==== Step 9: Report Best Parameters (Random Forest) ====

print("Best Parameters (Random Forest):", model.best_params_)

# ==== Step 10: Report Best Score (Random Forest) ====

# best_score_ is the mean cross-validated score of the best estimator.
print("Best Score (Random Forest):", model.best_score_)

# ==== Step 11: Report Best Estimator (Random Forest) ====

# best_estimator_ gives the actual model object with optimal hyperparameters.
print("Best Estimator (Random Forest):", model.best_estimator_)

# ============================================================
# Support Vector Classifier (SVC) Hyperparameter Tuning
# ============================================================

# ==== Step 12: Define Hyperparameter Search Space (SVC) ====

# Hyperparameter: kernel function
# Options: 'poly', 'sigmoid', 'linear', 'rbf'

hyperparameter1 = {'kernel': ['poly', 'sigmoid', 'linear', 'rbf']}
print("SVC hyperparameter search space defined:", hyperparameter1)

# ==== Step 13: Import SVC ====

# Support Vector Classifier is effective for high-dimensional spaces
# and uses kernel functions to separate classes.
from sklearn.svm import SVC

# ==== Step 14: Apply GridSearchCV (SVC) ====

model1 = GridSearchCV(SVC(), hyperparameter1)
model1.fit(X_train, y_train)
print("GridSearchCV fitting completed for SVC.")

# ==== Step 15: Report Best Estimator (SVC) ====
print("Best Estimator (SVC):", model1.best_estimator_)

# ==== Step 16: Report Best Parameters (SVC) ====
print("Best Parameters (SVC):", model1.best_params_)

# ==== Step 17: Report Best Score (SVC) ====
print("Best Score (SVC):", model1.best_score_)

# ============================================================
# Final Summary (Recruiter-Friendly Wrap-Up)
# ============================================================
#
# Hyperparameter Tuning Results:
# - RandomForestClassifier:
#   * Best Parameters: model.best_params_
#   * Best Score: model.best_score_
#   * Best Estimator: model.best_estimator_
#
# - Support Vector Classifier (SVC):
#   * Best Parameters: model1.best_params_
#   * Best Score: model1.best_score_
#   * Best Estimator: model1.best_estimator_
#
# Key Takeaway:
# Hyperparameter tuning ensures that models are optimized for performance.
# This systematic approach improves accuracy and generalization, making
# the model more reliable for real-world applications.
#
# Techniques Highlighted:
# - train_test_split: splitting data into training/testing sets
# - RandomForestClassifier: ensemble learning with decision trees
# - SVC: kernel-based classification
# - GridSearchCV: exhaustive search for best hyperparameters
#
# This script is self-documenting and recruiter-friendly, showing both
# technical workflow and explanatory narrative in one place.
# ============================================================