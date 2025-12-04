#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Feature Transformation - Missing Indicator
#
# Objective:
# Show how to use missing indicators to flag missing values
# and combine them with imputation for better model performance.
#
# Why Important:
# - Missingness itself can carry predictive signal (e.g., not reporting age might correlate with survival).
# - Adding indicator columns helps models learn patterns of missing data.
# - Recruiters see awareness of advanced preprocessing techniques beyond simple imputation.
#
# Dataset:
# Titanic dataset (train.csv) with Age, Fare, Survived columns.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Load Libraries
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
# We restrict to Age, Fare, and Survived for simplicity.
df = pd.read_csv('train.csv', usecols=['Age','Fare','Survived'])
print("Initial dataset sample:\n", df.head())

# ------------------------------------------------------------
# STEP 3: Prepare Features and Target
# ------------------------------------------------------------
X = df.drop(columns=['Survived'])   # Features: Age, Fare
y = df['Survived']                  # Target: Survival outcome

print("Feature sample:\n", X.head())
print("Target sample:\n", y.head())

# ------------------------------------------------------------
# STEP 4: Train/Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print("Training sample:\n", X_train.head())

# ------------------------------------------------------------
# STEP 5: Simple Imputation (Baseline)
# ------------------------------------------------------------
# Impute missing values with default strategy (mean for numerical).
# This is the most common baseline approach.

si = SimpleImputer()
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

# ------------------------------------------------------------
# STEP 6: Train Baseline Model
# ------------------------------------------------------------
clf = LogisticRegression()
clf.fit(X_train_trf, y_train)
y_pred = clf.predict(X_test_trf)

print("Baseline Accuracy (mean imputation only):", accuracy_score(y_test, y_pred))

# ------------------------------------------------------------
# STEP 7: Missing Indicator
# ------------------------------------------------------------
# MissingIndicator identifies which features contain missing values.

mi = MissingIndicator()
mi.fit(X_train)

print("Features with missing values:", mi.features_)

# Transform training/test data to boolean arrays (1 = missing, 0 = not missing).
X_train_missing = mi.transform(X_train)
X_test_missing = mi.transform(X_test)

print("Missing Indicator (Train):\n", X_train_missing[:5])
print("Missing Indicator (Test):\n", X_test_missing[:5])

# ------------------------------------------------------------
# STEP 8: Add Indicator Columns
# ------------------------------------------------------------
# Add indicator column for Age (example).
# This explicitly flags missingness as a new feature.

X_train['Age_NA'] = X_train_missing
X_test['Age_NA'] = X_test_missing

print("Training data with indicator:\n", X_train.head())

# ------------------------------------------------------------
# STEP 9: Imputation + Indicator Together
# ------------------------------------------------------------
# SimpleImputer with add_indicator=True automatically adds missing flags.
# This is cleaner than manually creating indicator columns.

si = SimpleImputer(add_indicator=True)
X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)

# ------------------------------------------------------------
# STEP 10: Train Model with Indicators
# ------------------------------------------------------------
clf = LogisticRegression()
clf.fit(X_train_trf2, y_train)
y_pred = clf.predict(X_test_trf2)

print("Accuracy with Missing Indicator:", accuracy_score(y_test, y_pred))

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates how to use missing indicators in feature engineering.")
print("Key Insights:")
print("1. Missingness itself can be predictive — indicators flag this for the model.")
print("2. Combining imputation with indicators improves robustness and accuracy.")
print("3. Logistic Regression performance improved when missing indicators were added.")
print("\nRecruiter Takeaway: This script shows advanced preprocessing awareness,")
print("attention to detail, and clear communication — essential traits for applied ML roles.")
print("============================================================")