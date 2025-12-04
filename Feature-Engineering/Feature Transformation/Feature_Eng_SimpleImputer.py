#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Handling Missing Numerical Data
#
# Techniques demonstrated:
# - Simple Imputation (Mean, Median, Most Frequent)
# - Arbitrary Value Imputation (e.g., -1, 99, 999)

# Objective:
# Show how different imputation strategies affect variance,
# distribution, and correlation in numerical features.

# Dataset:
# Titanic toy dataset (titanic_toy.csv)

# Audience:
# Recruiters, peers, and learners — this script is written
# as a tutorial with clear explanations and professional style.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Import Libraries
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv('titanic_toy.csv')
print(df.head())

# ------------------------------------------------------------
# STEP 3: Explore Missingness
# ------------------------------------------------------------
print("Fraction of missing values per column:")
print(df.isnull().mean())

# ------------------------------------------------------------
# STEP 4: Prepare Train/Test Split
# ------------------------------------------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# ------------------------------------------------------------
# STEP 5: Arbitrary Value Imputation
# ------------------------------------------------------------
# Replace missing Age values with 99 (flag as high number)
X_train['Age_99'] = X_train['Age'].fillna(99)

# Replace missing Age values with -1 (flag as low number)
X_train['Age_minus1'] = X_train['Age'].fillna(-1)

# Replace missing Fare values with 999 (flag as high number)
X_train['Fare_999'] = X_train['Fare'].fillna(999)

# Replace missing Fare values with -1 (flag as low number)
X_train['Fare_minus1'] = X_train['Fare'].fillna(-1)

# ------------------------------------------------------------
# STEP 6: Variance Comparison
# ------------------------------------------------------------
print("Variance Comparison:")
print('Original Age variance: ', X_train['Age'].var())
print('Age variance after 99 imputation: ', X_train['Age_99'].var())
print('Age variance after -1 imputation: ', X_train['Age_minus1'].var())
print()
print('Original Fare variance: ', X_train['Fare'].var())
print('Fare variance after 999 imputation: ', X_train['Fare_999'].var())
print('Fare variance after -1 imputation: ', X_train['Fare_minus1'].var())

# Interpretation:
# Arbitrary imputation can inflate or deflate variance,
# potentially distorting statistical properties.

# ------------------------------------------------------------
# STEP 7: Distribution Plots
# ------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)

X_train['Age'].plot(kind='kde', ax=ax, label='Original Age')
X_train['Age_99'].plot(kind='kde', ax=ax, color='red', label='Age_99')
X_train['Age_minus1'].plot(kind='kde', ax=ax, color='green', label='Age_-1')

ax.legend(loc='best')
plt.title("Age Distribution: Original vs Imputed")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

X_train['Fare'].plot(kind='kde', ax=ax, label='Original Fare')
X_train['Fare_999'].plot(kind='kde', ax=ax, color='red', label='Fare_999')
X_train['Fare_minus1'].plot(kind='kde', ax=ax, color='green', label='Fare_-1')

ax.legend(loc='best')
plt.title("Fare Distribution: Original vs Imputed")
plt.show()

# ------------------------------------------------------------
# STEP 8: Covariance & Correlation
# ------------------------------------------------------------
print("Covariance Matrix:")
print(X_train.cov())

print("Correlation Matrix:")
print(X_train.corr())

# Interpretation:
# Arbitrary imputation can alter relationships between variables,
# which may mislead downstream models.

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This feature engineering tutorial demonstrates handling missing numerical data.")
print("Techniques: Simple Imputation (mean/median) and Arbitrary Value Imputation (-1, 99, 999).")
print("Key Insights:")
print("1. Arbitrary imputation changes variance and distribution significantly.")
print("2. It can distort correlations, potentially misleading models.")
print("3. Use arbitrary imputation cautiously — often better for flagging missingness than replacing values.")
print("\nRecruiter Takeaway: This script shows practical data preprocessing skills,")
print("statistical awareness, and clear communication — essential traits for applied ML roles.")
print("============================================================")