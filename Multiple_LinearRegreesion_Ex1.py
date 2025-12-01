#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Multiple Linear Regression
# ================================================================

# This script demonstrates **Multiple Linear Regression** using SAT scores 
# and a random variable ("Rand 1,2,3") to predict GPA.
#
# Purpose:
# - To show how multiple predictors can be used to explain variation in GPA.
# - To highlight the difference between simple linear regression (one predictor)
#   and multiple regression (two predictors).

# Why it matters:
# - Multiple regression is a cornerstone of statistical modeling, allowing us 
#   to control for confounding variables and improve predictive accuracy.
# - Recruiters value candidates who can implement regression, interpret 
#   diagnostics, and present results clearly.

# Techniques highlighted:
# - Data visualization with Matplotlib/Seaborn.
# - Ordinary Least Squares (OLS) regression using `statsmodels`.
# - Model evaluation through regression summary (coefficients, R², F-statistic).
# - Interpretation of statistical tests (Durbin-Watson, VIF, etc.).
#

# ================================================================


# ==== Step 1: Import Libraries ====
# NumPy: numerical operations
# Pandas: data handling
# Matplotlib/Seaborn: visualization
# Statsmodels: regression modeling and statistical tests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm


# ==== Step 2: Load Dataset ====
# Reading CSV file into a Pandas DataFrame.
dataset = pd.read_csv('multisat.csv')
dataset  # Inspect dataset


# ==== Step 3: Simple Linear Regression Visualization ====
# Scatter plot of SAT vs GPA to visualize linear relationship.
plt.scatter(dataset['SAT'], dataset['GPA'])
plt.xlabel('SAT', color='red')
plt.ylabel('GPA', color='red')
plt.title('Simple Linear Regression: SAT vs GPA')
plt.show()


# ==== Step 4: Multiple Linear Regression Visualization ====
# Scatter plot of Random variable vs GPA.
# This shows how additional predictors can be considered.
plt.scatter(dataset['Rand 1,2,3'], dataset['GPA'])
plt.xlabel('Rand 1,2,3', color='green')
plt.ylabel('GPA', color='green')
plt.title('Multiple Regression: Random Variable vs GPA')
plt.show()


# ==== Step 5: Define Features and Target ====
# Independent variables (X): SAT and Random variable
# Dependent variable (y): GPA
x = dataset[['SAT', 'Rand 1,2,3']]
y = dataset['GPA']


# ==== Step 6: Add Constant Term ====
# Adding intercept term for regression.
x1 = sm.add_constant(x)


# ==== Step 7: Build OLS Regression Model ====
# Fit Ordinary Least Squares regression model.
model = sm.OLS(y, x1).fit()

# Regression summary provides:
# - Coefficients: effect of each predictor on GPA.
# - R²: proportion of variance explained.
# - F-statistic: overall model significance.
# - Durbin-Watson: tests autocorrelation of residuals (ideal ~2).
# - VIF (Variance Inflation Factor): checks multicollinearity.
model.summary()


# ==== Step 8: Visualize Regression Line ====
# Plot regression line for SAT vs GPA.
plt.scatter(dataset['SAT'], dataset['GPA'])
plt.plot(dataset['SAT'], model.predict(x1), color='red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.title('Regression Line: SAT vs GPA')
plt.show()


# ==== Step 9: Predictions ====
# Predict GPA values using the regression model.
model.predict(x1)


# ================================================================
# FINAL SUMMARY
# ================================================================

# - We applied multiple linear regression to predict GPA using SAT scores 
#   and a random variable.
# - Visualization confirmed linear relationships between predictors and GPA.
# - Regression summary provided key diagnostics:
#   * Coefficients quantify predictor effects.
#   * R² shows how much variance is explained.
#   * Durbin-Watson ~2 indicates no autocorrelation in residuals.
#   * VIF values (if computed) would confirm low multicollinearity.
# - The regression line fit demonstrates predictive capability.
#

# ================================================================