#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary : Feature Selection In ML
# ================================================================
# This script demonstrates **Feature Selection techniques in Machine Learning** using:
#   1. VarianceThreshold (filter method)
#   2. SelectFromModel (embedded method with LinearSVC)
#
# Feature selection is critical for:
# - Improving model performance by removing irrelevant/redundant features :   feature selection enhances the accuracy and generalizability of models. It helps the model focus on significant patterns in the data, reducing the risk of overfitting, which occurs when a model learns noise instead of meaningful patterns

# - Reducing training time and computational cost :   Fewer features lead to simpler models that require less computational power and time to train. This is particularly beneficial when dealing with large datasets, as it streamlines the learning process.

# - Mitigating the curse of dimensionality :  High-dimensional datasets can lead to overfitting, where models perform well on training data but poorly on unseen data. 

# - Enhancing interpretability of models : Simplifying models through feature selection makes them easier to interpret and understand.


# ================================================================


# ==== Step 1: Import Libraries ====

# NumPy: numerical operations
# Pandas: structured data handling
# Matplotlib & Seaborn: visualization
# sklearn.datasets: provides sample datasets (Iris used here)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# ==== Step 2: Load Dataset ====

# Iris dataset: 4 independent variables (features) and 3 dependent classes (labels).
pd.set_option('display.max_rows', None)  # Show full dataset if needed

Iris = load_iris()
Iris  # Displays dataset metadata

# Independent variables (features)
X = Iris.data

# Dependent variable (target classes: setosa, versicolor, virginica)
Y = Iris.target
Y


# ==== Step 3: VarianceThreshold Feature Selection ====

# VarianceThreshold removes features with low variance.
# Features with little variation across samples are unlikely to be useful predictors.
#  The primary goal is to reduce the dimensionality of the dataset by removing features that do not vary significantly across samples, which are unlikely to contribute meaningful information for predictive modeling.
#  The method calculates the variance for each feature in the dataset.
# It retains only those features whose variance is greater than or equal to the specified threshold (0.5 in this case).

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.5)  # Threshold set to 0.5
X_vt = sel.fit_transform(X)
X_vt

# Interpretation:
# - Features with variance < 0.5 are dropped.
# - This reduces dimensionality and focuses on informative features.


# ==== Step 4: Visualize Selected Features (3D Scatter Plot) ====

from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_vt[:,0], X_vt[:,1], X_vt[:,2], c=Y, cmap='rainbow')
ax.set_xlabel('X Axis') 
ax.set_ylabel('Y Axis')  
ax.set_zlabel('Z Axis')  
plt.show()


# ==== Step 5: Adjust VarianceThreshold (Threshold = 0.6) ====

# Increasing threshold removes more features, enforcing stricter selection.
sel1 = VarianceThreshold(threshold=0.6)
X_vt1 = sel1.fit_transform(X)
X_vt1

# Visualize distribution in 2D
plt.scatter(X_vt1[:,0], X_vt1[:,1], c=Y, cmap='rainbow')
plt.show()


# ==== Step 6: SelectFromModel Feature Selection ====

# SelectFromModel uses feature importance weights from a fitted model.
# Here, we use LinearSVC (Support Vector Classifier optimized for linear tasks).
# SelectFromModel allows you to select features based on their importance weights, which are derived from a fitted model.
# It helps in identifying the most relevant features for predictive modeling.
# By default, the threshold is set to the median importance of the features. You can also specify a custom threshold value.

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X, Y)

# Select features based on importance weights
sfm = SelectFromModel(model, prefit=True)
X_sfm = sfm.transform(X)
X_sfm

# Interpretation:
# - Features with weights above threshold are retained.
# - By default, threshold = median importance.
# - This method is adaptive and model-driven.


# ==== Step 7: Visualize Selected Features (3D Scatter Plot) ====
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_sfm[:,0], X_sfm[:,1], X_sfm[:,2], c=Y, cmap='rainbow')
ax.set_xlabel('X Axis') 
ax.set_ylabel('Y Axis')  
ax.set_zlabel('Z Axis')  
plt.show()


# ================================================================
# 📊 Statistical Test Interpretations (Recruiter-Friendly Notes)
# ================================================================
# While not directly applied here, these tests are critical in regression analysis:
#
# - Durbin-Watson Test:
#   Detects autocorrelation in residuals. Values ~2 indicate independence.
#
# - QQ Plot:
#   Compares residuals to normal distribution. Straight line → normality assumption holds.
#
# - Goldfeld-Quandt Test:
#   Tests for heteroscedasticity. Constant variance → assumption satisfied.
#
# - Variance Inflation Factor (VIF):
#   Detects multicollinearity. VIF > 10 indicates problematic correlation.
#
# Awareness of these tests demonstrates readiness for broader model validation.


# ================================================================
# 📌 Final Summary
# ================================================================
# - VarianceThreshold (filter method) removed low-variance features, simplifying dataset.
# - SelectFromModel (embedded method) used LinearSVC to select features based on importance weights.
# - Visualizations in 2D and 3D confirmed separation of classes after feature selection.
# - Awareness of statistical validation techniques (Durbin-Watson, QQ plot, Goldfeld-Quandt, VIF)
#   shows professional-level understanding of model assumptions.

# ================================================================