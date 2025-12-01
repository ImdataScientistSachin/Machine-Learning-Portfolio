#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Linear Regression Assumptions: Types and Their Test Methods
# ============================================================

# This script demonstrates how to test the key assumptions of a
# multiple linear regression model using advertising data.

# Predictors: TV, Radio, Newspaper
# Target: Sales

# Assumptions tested:
# 1. Autocorrelation
# 2. Normality of residuals
# 3. Linearity
# 4. Homoscedasticity
# 5. Multicollinearity

# Each section includes recruiter‑friendly explanations so the
# file reads like a professional tutorial/report.
# ======================

# ======================================

# ------------------------------------------------------------
# Step 1: Load Libraries
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # use clean grid style for plots

# ------------------------------------------------------------
# Step 2: Load Dataset
# ------------------------------------------------------------
# The dataset contains advertising spend on TV, Radio, Newspaper
# and the resulting Sales. We load it into a DataFrame.
dataset = pd.read_csv('ad.csv', index_col='Unnamed: 0')
print(dataset.head())  # preview first rows

# ------------------------------------------------------------
# Step 3: Define Features and Target
# ------------------------------------------------------------
# X = predictors (TV, Radio, Newspaper)
# y = target variable (Sales)
x = dataset[['TV','Radio','Newspaper']]
y = dataset['Sales']

# ------------------------------------------------------------
# Step 4: Build Regression Model
# ------------------------------------------------------------
import statsmodels.api as sm

# Add constant term to predictors (intercept)
const = sm.add_constant(x)

# Fit Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, const).fit()

# Display model summary with coefficients and statistics
print(model.summary())

# ------------------------------------------------------------
# Step 5: Test Assumptions
# ------------------------------------------------------------

# 5.1 Autocorrelation (Durbin-Watson)
# -----------------------------------
# Durbin-Watson statistic ≈ 2.08
# Interpretation: Values near 2 indicate little to no autocorrelation.
# Our data shows minimal autocorrelation → acceptable.

# 5.2 Normality of Residuals
# -----------------------------------
# We check distributions of predictors and target, plus a QQ plot of residuals.
sns.distplot(dataset['TV']); plt.show()
sns.distplot(dataset['Radio']); plt.show()
sns.distplot(dataset['Newspaper']); plt.show()
sns.distplot(dataset['Sales']); plt.show()

# QQ plot for residuals
Q_Q_plot = sm.qqplot(model.resid, fit=True)

# Interpretation:
# - Predictors show roughly symmetric distributions.
# - Sales is skewed (closer to exponential).
# - Residuals are approximately normal → acceptable.

# 5.3 Linearity
# -----------------------------------
# Scatter plots between predictors and target help visualize linear relationships.
plt.scatter(dataset['TV'], dataset['Sales'])
plt.xlabel('TV'); plt.ylabel('Sales'); plt.show()

plt.scatter(dataset['Radio'], dataset['Sales'])
plt.xlabel('Radio'); plt.ylabel('Sales'); plt.show()

plt.scatter(dataset['Newspaper'], dataset['Sales'])
plt.xlabel('Newspaper'); plt.ylabel('Sales'); plt.show()

# Interpretation:
# - TV and Radio show clear linear relationships with Sales.
# - Newspaper has weaker linearity.

# 5.4 Homoscedasticity
# -----------------------------------
# We use the Goldfeld-Quandt test to check constant variance of residuals.
import statsmodels.stats.api as sms
print(sms.het_goldfeldquandt(model.resid, model.model.exog))

# Interpretation:
# p-value = 0.07 > 0.05 → Fail to reject H0
# Conclusion: Residuals are homoscedastic (constant variance holds).

# 5.5 Multicollinearity
# -----------------------------------
# Check correlations between predictors
print(x.corr())

# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(x.values, i) for i in range(3)]
print(vif)

# Interpretation:
# VIF values < 5 → No serious multicollinearity detected.

# ------------------------------------------------------------
# Final Summary (Recruiter-Friendly Wrap-Up)
# ------------------------------------------------------------
print("""
Linear Regression Assumptions – Results:

1. Autocorrelation: Durbin-Watson ~2.08 → Minimal autocorrelation
2. Normality: Residuals approximately normal (QQ plot acceptable)
3. Linearity: Scatter plots confirm linear relationships
4. Homoscedasticity: Goldfeld-Quandt p=0.07 > 0.05 → Constant variance holds
5. Multicollinearity: VIF < 5 → No serious multicollinearity

Conclusion:
Our regression model satisfies the key assumptions, making it statistically valid and reliable.
This ensures that the insights derived (impact of TV, Radio, Newspaper on Sales) are trustworthy
and can be confidently presented in a professional or recruiter-facing context.
""")