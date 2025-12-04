#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Handling Missing Numerical Data - Arbitrary Value Imputation

# Definition:
# Arbitrary value imputation involves replacing missing values in a variable with a value that does not belong to the normal range of the variable, such as 0, -1, 999, or -999


# Objective:
# Demonstrate how arbitrary imputation affects variance, distribution,
# and correlation in numerical features.
#
# Why Important:
# - Quick way to flag missingness explicitly.
# - Preserves dataset size without dropping rows.
# - But can distort variance and correlations if misused.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Load Libraries
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
print("Initial dataset sample:\n", df.head())

# ------------------------------------------------------------
# STEP 3: Explore Missingness
# ------------------------------------------------------------
print("Fraction of missing values per column:\n", df.isnull().mean())
# INTERPRETATION: Age and Fare may contain missing values.
# Knowing this guides imputation strategy.

# ------------------------------------------------------------
# STEP 4: Prepare Features and Target
# ------------------------------------------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# ------------------------------------------------------------
# STEP 5: Arbitrary Value Imputation (Manual)
# ------------------------------------------------------------
# WHY: Replace missing values with constants outside normal range.
# WHAT: Use 99, -1 for Age; 999, -1 for Fare.
# HOW: .fillna() with chosen constants.
# Replace missing 'Age' values with 99

# This is useful if you want to flag missing values as a high number
X_train['Age_99'] = X_train['Age'].fillna(99)  # Impute missing Age with 99

# Replace missing 'Age' values with -1
# This is useful if you want to flag missing values as a low number
X_train['Age_minus1'] = X_train['Age'].fillna(-1)  # Impute missing Age with -1

# Replace missing 'Fare' values with 999
# This is useful if you want to flag missing values as a high number
X_train['Fare_999'] = X_train['Fare'].fillna(999)  # Impute missing Fare with 999

# Replace missing 'Fare' values with -1
# This is useful if you want to flag missing values as a low number
X_train['Fare_minus1'] = X_train['Fare'].fillna(-1)  # Impute missing Fare with -1

# ------------------------------------------------------------
# STEP 6: Variance Comparison
# ------------------------------------------------------------
print('Original Age variance: ', X_train['Age'].var())
print('Age variance after 99 imputation: ', X_train['Age_99'].var())
print('Age variance after -1 imputation: ', X_train['Age_minus1'].var())
print()
print('Original Fare variance: ', X_train['Fare'].var())
print('Fare variance after 999 imputation: ', X_train['Fare_999'].var())
print('Fare variance after -1 imputation: ', X_train['Fare_minus1'].var())
# INTERPRETATION: Arbitrary imputation inflates/deflates variance,
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
# INTERPRETATION: Arbitrary values create spikes in distribution,
# which may mislead models.

# ------------------------------------------------------------
# STEP 8: Covariance & Correlation
# ------------------------------------------------------------
print("Covariance Matrix:\n", X_train.cov())
print("Correlation Matrix:\n", X_train.corr())

# INTERPRETATION: Arbitrary imputation alters relationships between variables,
# potentially misleading downstream models.

# ------------------------------------------------------------
# STEP 9: Arbitrary Imputation with Scikit-Learn
# ------------------------------------------------------------

# WHY: Use SimpleImputer with constant strategy for reproducibility.
# WHAT: Replace Age with 99, Fare with 999.
# HOW: ColumnTransformer applies imputers to specific columns.

imputer1 = SimpleImputer(strategy='constant', fill_value=99)   # Age
imputer2 = SimpleImputer(strategy='constant', fill_value=999)  # Fare

trf = ColumnTransformer([
    ('imputer1', imputer1, [0]),  # Age column
    ('imputer2', imputer2, [1])   # Fare column
], remainder='passthrough')

trf.fit(X_train)

print("Imputer statistics (Age):", trf.named_transformers_['imputer1'].statistics_)
print("Imputer statistics (Fare):", trf.named_transformers_['imputer2'].statistics_)

X_train = trf.transform(X_train)
X_test = trf.transform(X_test)

print("Transformed Training Data Sample:\n", X_train[:5])
print("Transformed Test Data Sample:\n", X_test[:5])

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates Arbitrary Value Imputation for numerical features.")
print("Key Insights:")
print("1. Arbitrary imputation is quick but distorts variance and distributions.")
print("2. It can alter correlations, potentially misleading models.")
print("3. Best used to flag missingness explicitly, not as a statistical replacement.")
print("\nRecruiter Takeaway: This script shows practical preprocessing skills,")
print("awareness of statistical impacts, and clear communication — essential traits for applied ML roles.")
print("============================================================")