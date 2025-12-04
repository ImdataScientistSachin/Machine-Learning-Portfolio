#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Automatically Selecting Imputer Parameters
#
# Objective:
# Demonstrate how to automatically select the best imputation
# strategies for numerical and categorical features using
# GridSearchCV with a preprocessing + modeling pipeline.

# For numerical data, the default is typically the mean or interpolation (for time series).
# # For categorical data, the default is usually the mode.

# Why Important:
# - Shows awareness of preprocessing choices (mean vs median, mode vs constant).
# - Demonstrates integration of preprocessing with model training.
# - Highlights ability to tune hyperparameters systematically.
#
# Dataset:
# Titanic dataset (train.csv)
#
# Audience:
# Recruiters, peers, and learners — this script is written
# as a tutorial with clear explanations and professional style.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Import Libraries
# ------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import set_config

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv('train.csv')
print(df.head())

# ------------------------------------------------------------
# STEP 3: Clean Dataset
# ------------------------------------------------------------
# Drop unnecessary columns that don’t contribute to prediction
df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)
print("Dataset after dropping unnecessary columns:")
print(df.head())

# ------------------------------------------------------------
# STEP 4: Prepare Features and Target
# ------------------------------------------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

print("Training sample:")
print(X_train.head())

# ------------------------------------------------------------
# STEP 5: Define Preprocessing Pipelines
# ------------------------------------------------------------
# Numerical features
# 1. Impute missing values using the median (robust to outliers)
# 2. Standardize features by removing the mean and scaling to unit variance

numerical_features = ['Age', 'Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Default: median
    ('scaler', StandardScaler())                   # Standardize values
])

# Categorical features
categorical_features = ['Embarked', 'Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Default: mode
    ('ohe', OneHotEncoder(handle_unknown='ignore'))        # Encode categories
])

# Combine preprocessing
# Combine preprocessing steps for numerical and categorical features

preprocessor = ColumnTransformer(
    transformers=[
        # Apply the numerical_transformer pipeline to numerical_features
        ('num', numerical_transformer, numerical_features),
        # Apply the categorical_transformer pipeline to categorical_features
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------------------------
# STEP 6: Build Full Pipeline
# ------------------------------------------------------------
# Create a machine learning pipeline that combines preprocessing and classification

clf = Pipeline(steps=[
    # Step 1: Apply the preprocessor (handles both numerical and categorical features)
    ('preprocessor', preprocessor),
    # Step 2: Fit a logistic regression model to the preprocessed data
    ('classifier', LogisticRegression())
])

# Visualize pipeline structure as a diagram
set_config(display='diagram')
clf

# ------------------------------------------------------------
# STEP 7: Define Hyperparameter Grid
# ------------------------------------------------------------
param_grid = {
    # Try both 'mean' and 'median' strategies for imputing numerical features
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    
    # Try both 'most_frequent' and 'constant' strategies for imputing categorical features
    'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
    
    # Test different regularization strengths for Logistic Regression
    'classifier__C': [0.1, 1.0, 10, 100]
}

# ------------------------------------------------------------
# STEP 8: Grid Search Cross-Validation
# ------------------------------------------------------------
# Set up grid search with 10-fold cross-validation to find the best combination of parameters
grid_search = GridSearchCV(clf, param_grid, cv=10)

# Fit the grid search object to the training data
# This will search for the best combination of preprocessing and model parameters

grid_search.fit(X_train, y_train)

print("Best params:")
print(grid_search.best_params_)
print(f"Internal CV score: {grid_search.best_score_:.3f}")

# ------------------------------------------------------------
# STEP 9: Analyze Results
# ------------------------------------------------------------
# Convert the cross-validation results from grid search into a DataFrame for easy analysis
cv_results = pd.DataFrame(grid_search.cv_results_)

# Sort the results by the mean test score in descending order (best models first)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)

print("Top hyperparameter combinations:")

# Display the key hyperparameters and their corresponding mean test scores
print(cv_results[
    [
        'param_classifier__C',
        'param_preprocessor__cat__imputer__strategy',
        'param_preprocessor__num__imputer__strategy',
        'mean_test_score'
    ]
])

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates automatic selection of imputer parameters using GridSearchCV.")
print("Key Insights:")
print("1. Numerical features can be imputed with mean or median; categorical with mode or constant.")
print("2. GridSearchCV systematically tests preprocessing + model hyperparameters together.")
print("3. Logistic Regression performance improves when preprocessing choices are tuned.")
print("\nRecruiter Takeaway: This script shows strong ML engineering skills,")
print("awareness of preprocessing impacts, and ability to integrate tuning into pipelines.")
print("============================================================")