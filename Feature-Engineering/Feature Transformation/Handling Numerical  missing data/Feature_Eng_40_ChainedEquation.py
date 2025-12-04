#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Chained Equations for Missing Values (MICE)
#
# Definition:
# MICE (Multivariate Imputation by Chained Equations) generates multiple
# plausible imputations through iterative modeling. Each variable with
# missing data is imputed by regressing it on other variables in a cycle.


# Assumption:
# Operates under Missing At Random (MAR) — missingness depends only on observed data.
#
# Why Important:
# - Handles mixed data types (continuous, categorical, binary).
# - Preserves multivariate relationships.
# - Accounts for uncertainty in missing values.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Load Libraries
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
# WHY: Use 50_Startups dataset to demonstrate MICE.
# WHAT: Select relevant columns and scale down for simplicity.
# HOW: Divide by 10,000 and round values.

df = np.round(pd.read_csv('50_Startups.csv')[['R&D Spend','Administration','Marketing Spend','Profit']] / 10000)

# Set random seed for reproducibility
np.random.seed(9)

# Randomly sample 5 rows
df = df.sample(5)
print("Initial dataset sample:\n", df)

# ------------------------------------------------------------
# STEP 3: Drop Target Column
# ------------------------------------------------------------
# WHY: Focus on predictors only.
# WHAT: Remove 'Profit' column.

df = df.iloc[:,0:-1]
print("Predictor dataset:\n", df)

# ------------------------------------------------------------
# STEP 4: Introduce Missingness
# ------------------------------------------------------------
# WHY: Simulate missing values for demonstration.
# WHAT: Set specific cells to NaN.

# Set the value in the second row and first column to NaN (missing value)
df.iloc[1,0] = np.NaN   # R&D Spend missing
df.iloc[3,1] = np.NaN   # Administration missing
df.iloc[-1,-1] = np.NaN # Marketing Spend missing
print("Dataset with missing values:\n", df.head())

# ------------------------------------------------------------
# STEP 5: Initial Mean Imputation
# ------------------------------------------------------------
# WHY: Provide starting point for chained equations.
# WHAT: Fill missing values with column means.

df0 = pd.DataFrame()
df0['R&D Spend'] = df['R&D Spend'].fillna(df['R&D Spend'].mean())
df0['Administration'] = df['Administration'].fillna(df['Administration'].mean())
df0['Marketing Spend'] = df['Marketing Spend'].fillna(df['Marketing Spend'].mean())

print("0th Iteration (mean imputation):\n", df0)

# ------------------------------------------------------------
# STEP 6: Remove Imputed Value for Iteration
# ------------------------------------------------------------
# WHY: Begin chained equation cycle by re‑introducing missingness.
# WHAT: Set one imputed value back to NaN for prediction.

df1 = df0.copy()
df1.iloc[1,0] = np.NaN
print("Iteration dataset with missing R&D Spend:\n", df1)

# ------------------------------------------------------------
# STEP 7: Build Regression Model
# ------------------------------------------------------------
# WHY: Predict missing R&D Spend using Administration + Marketing Spend.
# WHAT: Use available rows (excluding missing).
X = df1.iloc[[0,2,3,4],1:3]  # predictors
y = df1.iloc[[0,2,3,4],0]    # target
print("Predictors:\n", X)
print("Target:\n", y)

# ------------------------------------------------------------
# STEP 8: Fit Model & Predict Missing Value
# ------------------------------------------------------------
lr = LinearRegression()
lr.fit(X,y)

predicted_value = lr.predict(df1.iloc[1,1:].values.reshape(1,2))
print("Predicted R&D Spend for missing row:", predicted_value)
# INTERPRETATION: MICE uses regression to iteratively estimate missing values,
# cycling through variables until convergence.

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates Multivariate Imputation by Chained Equations (MICE).")
print("Key Insights:")
print("1. MICE iteratively models each variable with missing data using other variables.")
print("2. Initial mean imputation provides a starting point for chained equations.")
print("3. Regression predicts missing values, preserving multivariate relationships.")
print("4. MICE accounts for uncertainty by generating multiple plausible imputations.")
print("\nRecruiter Takeaway: This script shows advanced preprocessing skills,")
print("awareness of statistical modeling, and clear communication — essential traits for applied ML roles.")
print("============================================================")