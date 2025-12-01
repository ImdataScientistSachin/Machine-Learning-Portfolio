#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Model Selection for Regression
# ================================================================
# This script demonstrates **Model Selection for Regression** using multiple  algorithms on the "Position_Salaries" dataset.

# Purpose:
# - To compare different regression techniques and evaluate their performance.
# - To highlight how model choice impacts predictive accuracy.

# Why it matters:
# - Recruiters value candidates who can not only implement models but also
#   critically evaluate which model is most appropriate for a given dataset.
# - This script showcases breadth of knowledge across regression techniques,
#   from linear models to ensemble methods.

# Techniques highlighted:
# - Data preprocessing with Pandas and NumPy.
# - Polynomial feature transformation for non-linear relationships.
# - Regression models: Logistic Regression (misapplied here, but included for demonstration),
#   Lasso, Ridge, Decision Tree, Gradient Boosting, Random Forest, KNN.
# - Model evaluation using train-test split and `.score()` method (R² metric).


# ================================================================


# ==== Step 1: Import Libraries ====
# NumPy: numerical operations
# Pandas: data handling
# Matplotlib/Seaborn: visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # Professional, clean plotting style


# ==== Step 2: Load Dataset ====
# Reading CSV file into a Pandas DataFrame.
dataset = pd.read_csv('Position_Salaries.csv')
dataset  # Inspect dataset


# ==== Step 3: Define Features and Target ====
# Independent variable (X): Position level
# Dependent variable (Y): Salary
# Reshape X into 2D array for scikit-learn compatibility.
X = dataset.iloc[:, 1].values
X = X.reshape(-1, 1)
X

Y = dataset.iloc[:, -1]
Y


# ==== Step 4: Import Regression Models ====
# Logistic Regression: classification algorithm, included here for demonstration.
# Lasso/Ridge: linear regression with regularization.
# PolynomialFeatures: transforms data to capture non-linear relationships.
# DecisionTree, GradientBoosting, RandomForest: tree-based ensemble methods.
# KNN: instance-based regression.
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# ==== Step 5: Polynomial Feature Transformation ====
# PolynomialFeatures generates higher-order terms to capture non-linear patterns.
degree = (1, 10)  # Degree of polynomial expansion
Poly = PolynomialFeatures(degree)
X_poly = Poly.fit_transform(X)
X_poly


# ==== Step 6: Prepare Models ====
# Each model is instantiated with parameters.
# Note: PolynomialFeatures is a transformer, not a predictive model.
models = [
    ('LR', LogisticRegression()),         # Logistic Regression (not ideal for regression tasks)
    ('LA', Lasso(alpha=5)),               # Lasso Regression (L1 regularization)
    ('RT', Ridge(alpha=5)),               # Ridge Regression (L2 regularization)
    ('DT', DecisionTreeRegressor()),      # Decision Tree Regression
    ('GB', GradientBoostingRegressor(n_estimators=5)),  # Gradient Boosting
    ('RF', RandomForestRegressor(n_estimators=5)),      # Random Forest
    ('KNN', KNeighborsRegressor(n_neighbors=5)),        # K-Nearest Neighbors
    ('PF', PolynomialFeatures(degree)),   # Transformer, not a model
]


# ==== Step 7: Train-Test Split ====
# Splitting data ensures unbiased evaluation.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Polynomial transformation applied to train/test sets
X_poly_train = Poly.fit_transform(X_train)
X_poly_test = Poly.transform(X_test)


# ==== Step 8: Model Training and Evaluation ====
# Loop through models, fit them, and evaluate performance using R² score.
for name, Cls in models:
    try:
        model = Cls  # Instantiate model/transformer

        if name in ['LR', 'LA', 'RT', 'DT', 'GB', 'RF', 'KNN']:
            # Fit using original features
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
        else:
            # PolynomialFeatures transformer (not predictive)
            model.fit(X_poly_train, y_train)
            score = model.score(X_poly_test, y_test)

        print(f"{name} Score: {score:.4f}")

    except Exception as e:
        print(f"Error fitting model {name}: {e}")


# ================================================================
# FINAL SUMMARY
# ================================================================

# - Multiple regression models were applied to the Position_Salaries dataset.
# - PolynomialFeatures allowed exploration of non-linear relationships.
# - R² scores provided a quick comparison of model performance.
# - Key insights:
#   * Regularized models (Lasso, Ridge) help prevent overfitting.
#   * Tree-based models (Decision Tree, Random Forest, Gradient Boosting) 
#     capture complex patterns and interactions.
#   * KNN regression relies on local neighborhood similarity.
#   * Logistic Regression is not appropriate for regression tasks, but its inclusion
#     demonstrates awareness of algorithm selection.
#

# ================================================================