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
# 3. Define model (RandomForestClassifier)
# 4. Specify hyperparameter search space
# 5. Apply GridSearchCV for tuning
# 6. Report best parameters

# ============================================================

# ==== Step 1: Load Libraries ====

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

# ==== Step 3: Transform Dataset ====

# Features (X) = measurements of iris flowers
# Target (Y) = species labels
X = Iris.data
Y = Iris.target
print("Feature matrix shape:", X.shape)
print("Target vector shape:", Y.shape)

# ==== Step 4: Train/Test Split ====

# Splitting the dataset ensures we can evaluate models on unseen data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)
print("Train/Test split completed.")

# ==== Step 5: Define Model ====

# We use RandomForestClassifier as the base model.
from sklearn.ensemble import RandomForestClassifier

# ==== Step 6: Import GridSearchCV ====

# GridSearchCV systematically searches through hyperparameter values
# using cross-validation to find the best configuration.
from sklearn.model_selection import GridSearchCV

# ==== Step 7: Define Hyperparameter Search Space ====

# Here we tune the number of trees (n_estimators) in the Random Forest.
hyperparameter = {'n_estimators': list(range(1, 20))}
print("Hyperparameter search space defined:", hyperparameter)

# ==== Step 8: Apply GridSearchCV ====

# GridSearchCV will train multiple models with different hyperparameters
# and select the one with the best performance.
model = GridSearchCV(RandomForestClassifier(), hyperparameter)
model.fit(X_train, y_train)
print("GridSearchCV fitting completed.")

# ==== Step 9: Report Best Parameters ====

# After training, we can access the best parameters found.
print("Best Parameters:", model.best_params_)

# ==== Final Summary ====
# ------------------------------------------------------------
# Hyperparameter Tuning Results:
# - We tuned RandomForestClassifier using GridSearchCV.
# - Search space: n_estimators from 1 to 19.
# - Best parameter identified: model.best_params_ (printed above).
#
# Key Takeaway:
# Hyperparameter tuning ensures that models are optimized for performance.
# This systematic approach improves accuracy and generalization, making
# the model more reliable for real-world applications.
# ------------------------------------------------------------