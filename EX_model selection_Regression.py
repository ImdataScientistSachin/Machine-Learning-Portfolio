#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Executive Summary: Model Selection in Regression
# ============================================================

# Purpose:
# This script demonstrates how to perform model selection for regression
# using multiple algorithms from scikit-learn. We compare models such as
# Logistic Regression, Lasso, Ridge, Decision Trees, Gradient Boosting,
# Random Forests, and KNN on a salary dataset.

# Why it matters:
# Recruiters and collaborators can see not only the code but also the
# reasoning behind each step. This makes the script a professional,
# tutorial-style report suitable for GitHub or LinkedIn.

# Dataset:
# Position_Salaries.csv — contains job positions and corresponding salaries.

# Workflow:
# 1. Load and preprocess data
# 2. Train/test split
# 3. Fit multiple regression models
# 4. Evaluate performance
# 5. Summarize results

# ============================================================

# ==== Step 1: Load Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # clean, professional plot style

import warnings
warnings.filterwarnings('ignore')  # suppress warnings for cleaner output

# ==== Step 2: Load Dataset ====

# The dataset contains job positions and their corresponding salaries.
# We load it into a DataFrame for analysis.
data = pd.read_csv('Position_Salaries.csv')
print(data.head())  # preview the dataset

# ==== Step 3: Define Features and Target ====
# X = predictor (Position level)
# Y = target variable (Salary)
X = data.iloc[:, 1].values
X = X.reshape(-1, 1)  # reshape for scikit-learn compatibility
print(X)

Y = data.iloc[:, 2].values
Y.reshape(-1, 1)
print(Y)

# ==== Step 4: Import Regression Models ====

# We import a variety of regression algorithms to compare performance.
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# ==== Step 5: Train/Test Split ====

# Splitting the dataset ensures we can evaluate models on unseen data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

# ==== Step 6: Prepare Models ====



# We create a list of models to test. Each model represents a different
# approach to regression, from linear to ensemble methods.
models = [
    ('LR', LogisticRegression()),          # Logistic Regression
    ('LA', Lasso(alpha=5)),                # Lasso Regression (L1 regularization)
    ('RT', Ridge(alpha=5)),                # Ridge Regression (L2 regularization)
    ('DT', DecisionTreeRegressor()),       # Decision Tree
    ('GB', GradientBoostingRegressor(n_estimators=5)),  # Gradient Boosting
    ('RF', RandomForestRegressor(n_estimators=5)),      # Random Forest
    ('KNN', KNeighborsRegressor(n_neighbors=5)),        # KNN Regression
    # ('PF', PolynomialFeatures(degree=5)),  # Transformer, not a model
]
print(models)

# ==== Step 7: Fit and Evaluate Models ====
# Loop through each model, fit it to the training data, and evaluate
# performance on the test set. Scores represent R^2 (coefficient of determination).
for name, Cls in models:
    model = Cls  # instantiate model
    model.fit(X_train, y_train)  # train model
    score = model.score(X_test, y_test)  # evaluate model
    print(name, ':', score)  # print model name and score

# ==== Step 8: Polynomial Features (Work in Progress) ====
# PolynomialFeatures is a transformer, not a model. It expands the feature
# space to capture non-linear relationships. Below is a draft workflow.
"""
Poly = PolynomialFeatures(degree=5)
X_poly = Poly.fit_transform(X)

for name, Cls in models:    
    try:
        if name == 'PF':
            # Instantiate PolynomialFeatures
            poly = Cls
            X_train_poly = poly.fit_transform(X)
            X_test_poly = poly.transform(X_test)
            continue  # skip fitting since PF is not a model
        
        # Instantiate the model
        model = Cls
        
        # Fit the model to training data
        model.fit(X_train_poly if name == 'PF' else X_train, y_train)
        
        # Evaluate on test data
        score = model.score(X_test_poly if name == 'PF' else X_test, y_test)
        
        print(name, ':', score)
        
    except Exception as e:
        print(f"Error fitting model {name}: {e}")
"""

# ==== Final Summary ====
# ------------------------------------------------------------
# Model Selection Results:
# - Multiple regression algorithms were tested on Position_Salaries data.
# - Each model provides a different perspective:
#   * Linear models (Lasso, Ridge) handle regularization.
#   * Tree-based models (Decision Tree, Random Forest, Gradient Boosting) capture non-linearities.
#   * KNN uses similarity-based predictions.
#
# Key Takeaway:
# This script demonstrates how to systematically compare regression models.
# The recruiter or collaborator can see both the technical workflow and
# the reasoning behind each choice, making it a polished, professional report.
# ------------------------------------------------------------