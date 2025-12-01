
#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : predicting ice cream unit sales based on temperature
# ================================================================
"""   ## IceCream_UnitSalesPrediction_ Modified    """  

# This script demonstrates predicting ice cream unit sales based on temperature
# using two regression techniques:
#   1. Linear Regression
#   2. Polynomial Regression

# In this modified version of the Ice Cream Unit Sales Prediction homework, we will explore both Linear Regression and Polynomial Regression techniques to predict ice cream sales based on temperature data. The dataset used contains temperature readings and corresponding ice cream sales units.


# Purpose:
# - To model the relationship between temperature and ice cream sales.
# - To compare linear vs polynomial regression in capturing non-linear patterns.

# Why It Matters (Recruiter-Friendly Narrative):
# - Regression analysis is a core skill in data science and business analytics.
# - Demonstrating both linear and polynomial regression shows ability to balance
#   simplicity vs complexity, interpret model fit, and visualize results.

# Key Concepts:
# - Linear Regression: Fits a straight line to data (simple, interpretable).
# - Polynomial Regression: Fits higher-order curves to capture complex patterns.
# - Bias-Variance Tradeoff: Higher polynomial degrees reduce bias but increase risk of overfitting.
# ================================================================


# ==== Step 1: Import Required Libraries ====
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')  # Professional plot styling


# ==== Step 2: Load Dataset ====
dataset = pd.read_csv('Ice_cream_selling_data.csv')
dataset

# Independent variable (Temperature)
X = dataset.iloc[:, [0]].values
X.round(3)
X.shape

# Dependent variable (Ice Cream Sales)
Y = dataset.iloc[:, [1]].values
Y.round(3)
Y.shape


# ==== Step 3: Visualize Raw Data ====
plt.scatter(X, Y)
plt.xlabel('Temperature (°C) → Independent Variable')
plt.ylabel('Ice Cream Sales (units) → Dependent Variable')
plt.title('Scatter Plot of Temperature vs Ice Cream Sales')
plt.show()


# ==== Step 4: Linear Regression ====
from sklearn.linear_model import LinearRegression

# Prepare and fit model
model = LinearRegression()
model.fit(X, Y)

# Evaluate model fit (R² score)
model.score(X, Y)

# Test prediction with manual input
test = np.array([[-6.316]])  # Example temperature
model.predict(test)

# Visualize linear regression line
plt.scatter(dataset['Temperature (°C)'], dataset['Ice Cream Sales (units)'])
plt.plot(X, model.predict(X), color='purple')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales (units)')
plt.title('Linear Regression Fit')
plt.show()


# ==== Step 5: Polynomial Regression ====
from sklearn.preprocessing import PolynomialFeatures

# Polynomial degree controls complexity
Poly = PolynomialFeatures(degree=7)

"""
# In the context of polynomial regression, the degree refers to the highest power of the independent variable in the polynomial equation. 
#It determines the level of complexity of the model.

# Degree 1 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋
# Degree 2 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋 + 𝛽2𝑋2 
# Degree 3 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋 + 𝛽2𝑋2 + 𝛽3𝑋3

# As the degree increases, 
# the polynomial can capture more complex patterns in the data. However, 
# higher degrees can also lead to overfitting, where the model fits the training data too closely and may not generalize well to unseen data.

# fit_transform: Learns the transformation from the data and applies it.
# It takes the input data and generates polynomial features based on the specified degree.

"""

# Transform features into polynomial terms
X_poly = Poly.fit_transform(X)
# transform: Applies the learned transformation to the data.
# It takes the input data and generates polynomial features based on the specified degree.

# Fit polynomial regression model
model1 = LinearRegression()
model1.fit(X_poly, Y)

# Evaluate model fit (R² score)
model1.score(X_poly, Y).round(3)

# Visualize polynomial regression curve
plt.scatter(dataset['Temperature (°C)'], dataset['Ice Cream Sales (units)'])
plt.plot(X, model1.predict(X_poly), color='green')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales (units)')
plt.title('Polynomial Regression Fit (Degree=7)')
plt.show()


# ==== Step 6: Test Polynomial Model Predictions ====
# Transform test input
test1 = Poly.transform(test)
model1.predict(test1).round(3)

# Manual test with another temperature
raw_test2 = np.array([[3.704057]])
test2 = Poly.transform(raw_test2)
model1.predict(test2).round(3)

# High-resolution curve for smooth visualization
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
plt.scatter(dataset['Temperature (°C)'], dataset['Ice Cream Sales (units)'])
plt.plot(X_grid, model1.predict(Poly.transform(X_grid)), color='orange')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales (units)')
plt.title('Polynomial Regression Curve (Degree=7)')
plt.show()


# ================================================================
# FINAL SUMMARY
# ================================================================
# Key Takeaways:
# - Linear regression provides a simple, interpretable baseline model.
# - Polynomial regression (degree=7) captures complex non-linear relationships
#   between temperature and ice cream sales.
# - Higher polynomial degrees improve fit but risk overfitting (poor generalization).
#
# Professional Impact:
# - Demonstrates ability to apply regression techniques, evaluate model fit,
#   and visualize results.
# - Highlights understanding of bias-variance tradeoff and model complexity.

# ================================================================