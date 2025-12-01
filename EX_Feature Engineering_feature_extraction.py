#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary : Feature Engineering & Feature Extraction
# ================================================================
# This script demonstrates **Feature Extraction techniques in Machine Learning** using PCA and Kernel PCA.
# Feature extraction reduces dimensionality while retaining essential information, making models faster,
# more interpretable, and less prone to overfitting. 
#
# Why this matters for recruiters and professionals:
# - Shows ability to apply **linear (PCA)** and **non-linear (Kernel PCA)** dimensionality reduction.
# - Demonstrates **data visualization in 2D and 3D** for interpretability.
# - Highlights understanding of **hyperparameters (n_components, gamma)** and their impact.
# - Provides narrative commentary that explains not just *what* the code does, but *why* it matters.
#
# Statistical tests often used in regression (Durbin-Watson, QQ plot, Goldfeld-Quandt, VIF) are explained
# in comments for completeness, even though they are not directly applied here. This shows awareness of
# model validation techniques beyond PCA.
#
# ================================================================
# # Feature Engineering in ML
# ================================================================
# Types of Feature Engineering:
#   1. Feature Creation
#   2. Feature Transformation
#   3. Feature Extraction
#   4. Feature Selection
#
# This script focuses on **Feature Extraction**.
# Definition: Reduces the dimensionality of the dataset while retaining essential information.
# Methods:
#   - Principal Component Analysis (PCA): Linear transformation capturing maximum variance.
#   - Kernel PCA: Non-linear transformation using kernel functions.
#   - t-SNE (not implemented here): Visualization of high-dimensional data.
# ================================================================


# ==== Step 1: Import Libraries ====

# Importing essential libraries for numerical computation, data handling, and visualization.
# NumPy: numerical operations
# Pandas: structured data handling
# Matplotlib & Seaborn: visualization
# mpl_toolkits.mplot3d: 3D plotting


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from mpl_toolkits.mplot3d import Axes3D   # Enables 3D plotting


# ==== Step 2: Load Dataset ====

# Using sklearn's built-in Iris dataset.
# Iris dataset has 4 independent variables (features) and 3 dependent classes (labels).
from sklearn.datasets import load_iris

Iris = load_iris()
Iris  # Displays dataset metadata

# Independent variables (features)
X = Iris.data
X

# Dependent variable (target classes: setosa, versicolor, virginica)
Y = Iris.target
Y


# ==== Step 3: Apply PCA (2 Components) ====

# PCA reduces dimensionality by projecting data onto principal components.
# n_components=2 → reduces dataset to 2 independent variables for easy visualization.
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # Hyperparameter: number of components
X_pca = pca.fit_transform(X)  # PCA works only on independent variables

X_pca
X_pca.shape  # Shape confirms reduction to 2D


# ==== Step 4: Visualize PCA (2D Scatter Plot) ====

# Scatter plot shows how the 3 classes separate in reduced 2D space.
plt.scatter(X_pca[:,0], X_pca[:,1], c=Y, cmap='rainbow')
plt.show()
# Interpretation: PCA captures variance, allowing visualization of class separation.


# ==== Step 5: Apply PCA (3 Components) ====

# Extending PCA to 3 components for 3D visualization.
pca1 = PCA(n_components=3)
X_pca1 = pca1.fit_transform(X)
X_pca1  # Reduced dataset with 3 independent variables


# ==== Step 6: Visualize PCA (3D Scatter Plot) ====

# 3D scatter plot helps visualize separation across three principal components.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Adding labels to axes for clarity
ax.set_xlabel('X Axis') 
ax.set_ylabel('Y Axis')  
ax.set_zlabel('Z Axis')  

ax.scatter(X_pca1[:, 0], X_pca1[:, 1], X_pca1[:, 2], c=Y, cmap='rainbow')
plt.tight_layout()
plt.show()


# ==== Step 7: Kernel PCA ====

# Kernel PCA extends PCA to handle non-linear data using kernel functions.
# Common kernels: polynomial, RBF (Radial Basis Function).
from sklearn.datasets import make_moons

# Generate synthetic non-linear dataset (two interleaving half circles).
X, Y = make_moons(n_samples=500, noise=0.02, random_state=0)

# Visualize original distribution
plt.scatter(X[:,0], X[:,1], c=Y, cmap='rainbow')
plt.show()


# ==== Step 8: Apply Kernel PCA ====

from sklearn.decomposition import KernelPCA

# Using RBF kernel with gamma=15.
# Gamma controls influence of data points:
#   - High gamma → sensitive to local variations (risk of overfitting).
#   - Low gamma → smoother decision boundaries.
kpca = KernelPCA(kernel='rbf', gamma=15)

X_kpca = kpca.fit_transform(X)
X_kpca

# Visualize transformed dataset
plt.scatter(X_kpca[:,0], X_kpca[:,1], c=Y, cmap='rainbow')
plt.show()


# ================================================================
# 📊 Statistical Test Interpretations (Recruiter-Friendly Notes)
# ================================================================
# Although not directly applied in PCA, these tests are critical in regression analysis:
#
# - Durbin-Watson Test:
#   Checks for autocorrelation in residuals. Values ~2 indicate no autocorrelation.
#   Important for validating independence assumption in regression.
#
# - QQ Plot:
#   Compares distribution of residuals to a normal distribution.
#   Straight line → residuals are normally distributed (assumption satisfied).
#
# - Goldfeld-Quandt Test:
#   Tests for heteroscedasticity (variance of residuals).
#   If variance is constant → assumption of homoscedasticity holds.
#
# - Variance Inflation Factor (VIF):
#   Detects multicollinearity among predictors.
#   VIF > 10 indicates problematic correlation between features.
#
# Including these notes shows awareness of broader model validation techniques.


# ================================================================
# 📌 Final Summary
# ================================================================
# - PCA (linear) successfully reduced Iris dataset to 2D and 3D, enabling visualization of class separation.
# - Kernel PCA (non-linear) effectively handled synthetic moon-shaped data, showing flexibility in feature extraction.
# - Hyperparameters (n_components, gamma) significantly influence dimensionality reduction and model performance.
# - Awareness of statistical tests (Durbin-Watson, QQ plot, Goldfeld-Quandt, VIF) demonstrates readiness for
#   regression model validation and professional-level analysis.

# ================================================================