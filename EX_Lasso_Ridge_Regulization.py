#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary : Lasso and Ridge
# ================================================================

# This script demonstrates **Regularization techniques (Lasso and Ridge)** in linear regression.
# Regularization helps prevent overfitting by penalizing large coefficients, encouraging simpler,
# more generalizable models.
#
# - **Ridge Regression (L2 penalty):** Shrinks coefficients by penalizing their squared magnitude.
# - **Lasso Regression (L1 penalty):** Shrinks coefficients by penalizing their absolute magnitude,
#   and can drive some coefficients to zero (feature selection).
#
# Recruiter-Friendly Takeaway:
# This script shows practical application of **regularization methods** with clear visualizations,
# highlighting awareness of model complexity, bias-variance tradeoff, and interpretability.
# ================================================================


# ==== Step 1: Import Libraries ====
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# ==== Step 2: Prepare Dataset ====

# Independent variable: Education level
edu = np.array([1, 2, 3, 4, 5]).reshape(5, 1)

# Dependent variable: Salary
salary = np.array([200, 300, 400, 500, 600]).reshape(5, 1)

# Visualize raw data
plt.scatter(edu, salary)
plt.xlabel('Education (Independent Variable)')
plt.ylabel('Salary (Dependent Variable)')
plt.title('Education vs Salary Distribution')
plt.show()


# ==== Step 3: Linear Regression (Baseline) ====

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(edu, salary)

print("Linear Regression R² Score:", model.score(edu, salary))

# Visualize linear fit
plt.scatter(edu, salary)
plt.plot(edu, model.predict(edu), color='purple')
plt.xlabel('Education')
plt.ylabel('Salary')
plt.title('Linear Regression Fit')
plt.show()

# Interpretation:
# Linear regression fits perfectly here, but in real-world scenarios this can lead to overfitting.


# ==== Step 4: Lasso Regression (L1 Regularization) ====
from sklearn.linear_model import Lasso

model1 = Lasso(alpha=35)  # alpha controls penalty strength
model1.fit(edu, salary)

print("Lasso Regression R² Score:", model1.score(edu, salary))

# Visualize Lasso fit
plt.scatter(edu, salary)
plt.plot(edu, model1.predict(edu), color='green')
plt.xlabel('Education')
plt.ylabel('Salary')
plt.title('Lasso Regression Fit (L1 Penalty)')
plt.show()

# Interpretation:
# Lasso tilts the regression line slightly, showing how L1 penalty shrinks coefficients.
# In larger datasets, Lasso can eliminate irrelevant features entirely.


# ==== Step 5: Ridge Regression (L2 Regularization) ====

from sklearn.linear_model import Ridge

model2 = Ridge(alpha=5)  # alpha controls penalty strength
model2.fit(edu, salary)

print("Ridge Regression R² Score:", model2.score(edu, salary))

# Visualize Ridge fit
plt.scatter(edu, salary)
plt.plot(edu, model2.predict(edu), color='red')
plt.xlabel('Education')
plt.ylabel('Salary')
plt.title('Ridge Regression Fit (L2 Penalty)')
plt.show()

# Interpretation:
# Ridge regression shrinks coefficients smoothly without eliminating them.
# It is effective when many features contribute to the outcome.


# ================================================================
# 📊 Conceptual Notes
# ================================================================
# - **Overfitting:** Model fits training data too closely, fails to generalize.
# - **Bias-Variance Tradeoff:** Regularization increases bias slightly but reduces variance, improving generalization.
# - **Alpha Parameter:** Controls penalty strength. Higher alpha → stronger shrinkage.
# - **Lasso vs Ridge:**
#   * Lasso (L1): Feature selection, sparse models.
#   * Ridge (L2): Coefficient shrinkage, stable models.

# Awareness of these tradeoffs demonstrates professional-level understanding of regression techniques.


# ================================================================
# 📌 Final Summary
# ================================================================
# - Linear regression fit the dataset perfectly but risked overfitting.
# - Lasso (L1) introduced penalty, tilting the regression line and encouraging sparsity.
# - Ridge (L2) introduced penalty, shrinking coefficients smoothly.
# - Both methods improve generalization and prevent overfitting in real-world scenarios.
#

# ================================================================