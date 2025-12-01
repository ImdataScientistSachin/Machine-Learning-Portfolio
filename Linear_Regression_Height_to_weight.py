#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Linear_Regression_Height_Convertor 
# ================================================================

# This script demonstrates a simple Linear Regression model to predict
# weight from height measurements.

# Purpose:
# - To model the linear relationship between height (independent variable)
#   and weight (dependent variable).
# - To showcase regression analysis using statsmodels and manual calculation.

# Why It Matters (Recruiter-Friendly Narrative):
# - Linear regression is one of the most fundamental predictive modeling techniques.
# - It is widely applied in health analytics, economics, and business forecasting.
# - Demonstrating regression with clear interpretation highlights ability to
#   apply statistical methods and communicate results effectively.

# Key Concepts:
# - Regression Line: y = β0 + β1X
# - β0 (Intercept): Expected weight when height = 0 (baseline).
# - β1 (Slope): Change in weight for each unit increase in height.
# ================================================================


# ==== Step 1: Import Required Libraries ====
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set_style('whitegrid')  # Professional plot styling


# ==== Step 2: Prepare Dataset ====
Height = np.array([62,62,63,65,66,67,68,68,70,72])  # Independent variable
Weight = np.array([120,140,130,150,142,130,135,175,149,168])  # Dependent variable

# Quick visualization of raw data
plt.scatter(Height, Weight)
plt.xlabel('Height (inches) → Independent Variable')
plt.ylabel('Weight (pounds) → Dependent Variable')
plt.title('Scatter Plot of Height vs Weight')
plt.show()


# ==== Step 3: Fit Linear Regression Model (Statsmodels) ====
# Add constant term for intercept
model = sm.add_constant(Height)

# Fit Ordinary Least Squares (OLS) regression
x = sm.OLS(Weight, model).fit()

# Display regression summary (coefficients, R², p-values)
x.summary()

# Predict weight for a given height (example: 72 inches)
x.predict(np.array([1.0, 72]))


# ==== Step 4: Manual Regression Equation ====

# Regression equation derived from fitted model:
# Weight = -78.3 + 3.3 * Height
prediction = -78.3 + 3.3 * Height

# Visualize regression line
plt.scatter(Height, Weight)
plt.plot(Height, prediction, color='green')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.title('Linear Regression Fit')
plt.show()


# ==== Step 5: Predictions for Specific Heights ====

# Predicting weight for different heights using regression equation
predicted_weight_70 = -78.3 + 3.3 * 70
print(f"The predicted weight for a height of 70 inches is: {predicted_weight_70:.1f} pounds")

predicted_weight_65 = -78.3 + 3.3 * 65
print(f"The predicted weight for a height of 65 inches is: {predicted_weight_65:.1f} pounds")

predicted_weight_68 = -78.3 + 3.3 * 68
print(f"The predicted weight for a height of 68 inches is: {predicted_weight_68:.1f} pounds")

predicted_weight_75 = -78.3 + 3.3 * 75
print(f"The predicted weight for a height of 75 inches is: {predicted_weight_75:.1f} pounds")


# ================================================================
# FINAL SUMMARY
# ================================================================

# Key Takeaways:
# - Linear regression successfully modeled the relationship between height and weight.
# - Slope (3.3) indicates that each additional inch of height increases weight by ~3.3 pounds.
# - Intercept (-78.3) is the baseline when height = 0 (not meaningful physically, but required mathematically).
#
# Professional Impact:
# - Demonstrates ability to apply regression analysis, interpret coefficients,
#   and make predictions.
# - Highlights statistical literacy and clear communication of results.

# ================================================================