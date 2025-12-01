#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary :  Polynomial regression
# ================================================================

# This script demonstrates **Polynomial Regression**, a technique that models the relationship between
# an independent variable (X) and a dependent variable (y) as an nth-degree polynomial.

# Polynomial regression is a type of regression analysis where the relationship between the independent variable 
# 𝑋  and the dependent variable 𝑦 is modeled as an 𝑛th degree polynomial. 
# In simpler terms, it allows for more flexibility by fitting a polynomial curve rather than a straight line to the data.

###  Formula  = 𝑦=𝛽0+𝛽1𝑋 + 𝛽2𝑋2 + 𝛽3𝑋3

# # How Polynomial Regression Works:
# 1. **Transform Features:** In polynomial regression, you create new features by raising the original features to various powers (e.g., \(X\), \(X^2\), \(X^3\), etc.).
# 2. **Fit the Model:** You then fit a linear regression model to these transformed features.
# 3. **Predict:** Use the fitted polynomial model to make predictions.

# Why Polynomial Regression matters:
# - Captures **non-linear relationships** that linear regression cannot.
# - Provides flexibility in fitting curves to complex datasets.
# - Useful in real-world scenarios like salary prediction, growth curves, and trend analysis.
#
# Recruiter-Friendly Takeaway:
# This script shows practical application of **polynomial regression** with clear visualizations,
# highlighting awareness of model complexity, overfitting risks, and interpretability.
# ================================================================


# ==== Step 1: Import Libraries ====

# NumPy: numerical operations
# Pandas: structured data handling
# Matplotlib & Seaborn: visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# ==== Step 2: Load Dataset ====

# Using Position_Salaries.csv dataset.
# Dataset contains "Level" (independent variable) and "Salary" (dependent variable).
dataset = pd.read_csv('Position_Salaries.csv')
dataset


# ==== Step 3: Visualize Raw Data ====

# Scatter plot shows distribution of Level vs Salary.
# Observation: Relationship is clearly non-linear, motivating polynomial regression.
plt.scatter(dataset['Level'], dataset['Salary'])
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.show()


# ==== Step 4: Prepare Features and Target ====

X = dataset.iloc[:, [1]].values  # Independent variable
Y = dataset.iloc[:, 2].values    # Dependent variable


# ==== Step 5: Fit Linear Regression (Baseline) ====

# Linear regression assumes straight-line relationship.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, Y)
model.score(X, Y)  # R² score (goodness of fit)

# Test prediction with manual input
test = np.array([[6.5]])  # Must be 2D array
model.predict(test)

# Visualize linear fit
plt.scatter(dataset['Level'], dataset['Salary'])
plt.plot(X, model.predict(X), color='purple')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.show()

# Interpretation:
# Linear regression fails to capture the curve in data → motivates polynomial regression.


# ==== Step 6: Polynomial Regression ====

# Polynomial regression creates new features (X², X³, …).
from sklearn.preprocessing import PolynomialFeatures

Poly = PolynomialFeatures(degree=5)  # Degree controls complexity
X_poly = Poly.fit_transform(X)       # Transform features into polynomial terms

# In the context of polynomial regression, the degree refers to the highest power of the independent variable in the polynomial equation. 
#It determines the level of complexity of the model.

# Degree 1 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋
# Degree 2 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋 + 𝛽2𝑋2 
# Degree 3 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋 + 𝛽2𝑋2 + 𝛽3𝑋3

# As the degree increases, 
# the polynomial can capture more complex patterns in the data. However, 
# higher degrees can also lead to overfitting, where the model fits the training data too closely and may not generalize well to unseen data.




# Fit polynomial regression model
model1 = LinearRegression()
model1.fit(X_poly, Y)
model1.score(X_poly, Y)  # Higher R² indicates better fit

# as the [default degree = 1 ] in the above value shown the anaccurate value
# so we can increase the value of degree till the Preddicted value will be accurate.
# degree = 5 


# ==== Step 7: Visualize Polynomial Fit ====

plt.scatter(dataset['Level'], dataset['Salary'])
plt.plot(X, model1.predict(X_poly), color='green')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.show()

# Interpretation:
# Polynomial regression curve fits the data much better than linear regression.


# ==== Step 8: Test Predictions ====

# Predict salary for Level = 6.5
test1 = Poly.transform(test)
model1.predict(test1)

# Predict salary for Level = 7.5
raw_test2 = np.array([[7.5]])
test2 = Poly.transform(raw_test2)
model1.predict(test2)


# ================================================================
# 📊 Statistical & Conceptual Notes (Recruiter-Friendly)
# ================================================================
# - Degree of Polynomial:
#   Controls complexity. Higher degree → more flexibility but risk of overfitting.
#
# - Overfitting:
#   Model fits training data too closely, may fail on unseen data.
#
# - Bias-Variance Tradeoff:
#   Low degree → high bias (underfitting).
#   High degree → high variance (overfitting).
#
# - Model Validation:
#   Use cross-validation and metrics (R², RMSE) to select optimal degree.
#
# - Related Statistical Tests:
#   * Durbin-Watson: checks autocorrelation in residuals.
#   * QQ Plot: checks normality of residuals.
#   * Goldfeld-Quandt: checks heteroscedasticity.
#   * VIF: checks multicollinearity in predictors.
#   These are not applied here but demonstrate awareness of regression diagnostics.


# ================================================================
# 📌 Final Summary
# ================================================================
# - Linear regression failed to capture the non-linear salary trend.
# - Polynomial regression (degree=5) successfully modeled the curve, improving fit.
# - Predictions for Levels 6.5 and 7.5 aligned more closely with actual salaries.
# - Awareness of overfitting and statistical validation shows professional-level understanding.

# ================================================================