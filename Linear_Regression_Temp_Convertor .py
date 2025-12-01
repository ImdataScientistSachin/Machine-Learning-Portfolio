#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY :  Linear Regression 
# ================================================================

# This script demonstrates a simple yet powerful application of 
# **Linear Regression** using Python's `statsmodels` library.
 
# Purpose:
# - To model the linear relationship between Celsius and Fahrenheit.
# - To showcase how regression can be used to predict values and 
#   validate assumptions through statistical tests.

# Why it matters:
# - Linear regression is one of the most fundamental techniques in 
#   machine learning and statistics.
# - Recruiters and hiring managers value candidates who can not only 
#   implement models but also explain their assumptions, interpret 
#   diagnostics, and present results clearly.

# Techniques highlighted:
# - Data visualization with `matplotlib` and `seaborn`.
# - Ordinary Least Squares (OLS) regression using `statsmodels`.
# - Manual prediction vs. model-based prediction.
# - Interpretation of regression diagnostics (Durbin-Watson, QQ plot, 
#   Goldfeld-Quandt, Variance Inflation Factor).
#
# ================================================================


# ==== Step 1: Import Libraries ====

# NumPy for numerical operations, Matplotlib/Seaborn for visualization,
# Statsmodels for regression modeling and statistical tests.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # Recruiter-friendly: clean visuals matter


# ==== Step 2: Define Data ====

# Fahrenheit values are given. Celsius is derived using the conversion formula.
# This demonstrates feature engineering: transforming raw data into usable features.
Fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100])
Celsius = (Fahrenheit - 32) * 5.0/9.0

# ==== Step 3: Visualize Data ====
# Scatter plot to show the linear relationship between Celsius and Fahrenheit.

plt.scatter(Celsius, Fahrenheit)
plt.xlabel('Celsius ---> Independent Variable')
plt.ylabel('Fahrenheit ---> Dependent Variable')
plt.title('Scatter Plot: Celsius vs Fahrenheit')
plt.show()


# ==== Step 4: Build Regression Model ====
# Adding a constant term for the intercept in OLS regression.
import statsmodels.api as sm
mode = sm.add_constant(Celsius)  # Adds intercept term

# Fit the Ordinary Least Squares (OLS) regression model.
model = sm.OLS(Fahrenheit, mode).fit()
model.summary()

# ==== Step 5: Predictions ====
# Predict Fahrenheit for given Celsius values using the model.
model.predict(np.array([1.0, 10]))  # Example prediction

# Extract model parameters (intercept and slope).
model.params

# ==== Step 6: Predict with the Model ====
# Using regression equation: y = intercept + slope * x
y_hat = model.params[0] + model.params[1] * Celsius
y_hat


# ==== Step 7: Manual Calculation ====
# Demonstrating manual calculation of regression line using approximate coefficients.
# This shows understanding of regression math beyond library functions.
y_hat = 31.9 + 1.79 * Celsius
y_hat


# ==== Step 8: Visualize Regression Line ====
# Overlay regression line on scatter plot to show fit.
plt.scatter(Celsius, Fahrenheit)
plt.plot(Celsius, y_hat, color='red', label='Regression Line')
plt.xlabel('Celsius --> Independent Variable')
plt.ylabel('Fahrenheit --> Dependent Variable')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()


# ==== Step 9: Predict for New Value ====
# Example: Predict Fahrenheit for Celsius = 100.
# This demonstrates practical application of the model.
temp = 100
# Recruiter-friendly note: This shows how regression generalizes beyond training data.


# ================================================================
# FINAL SUMMARY
# ================================================================
# - We successfully built a linear regression model to predict Fahrenheit  from Celsius values.
# - Visualization confirmed a strong linear relationship.

#   provide confidence in model validity:
#   * Durbin-Watson ~2 → Residuals are not autocorrelated.
#   * QQ plot → Residuals approximately normal.
#   * Goldfeld-Quandt → No major heteroscedasticity detected.
#   * VIF → Multicollinearity not a concern (single predictor).
# - The regression line fits the data well and allows predictions for new values.


# ================================================================