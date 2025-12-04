#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: End-to-End Dataset Processing with Pipelines
#
# Objective:
# Demonstrate how to build a full machine learning pipeline
# that handles missing values, encodes categorical features,
# scales numerical features, selects important features,
# trains a classifier, and evaluates performance.
#
# Why Important:
# - Pipelines ensure reproducibility and cleaner code.
# - They prevent data leakage by applying transformations consistently.
# - They make deployment easier (single object for preprocessing + model).
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

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv('train.csv')
print(df.head())

# ------------------------------------------------------------
# STEP 3: Prepare Dataset
# ------------------------------------------------------------
# Drop unnecessary columns
df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)
print("Dataset shape after dropping columns:", df.shape)
print(df.head())

# ------------------------------------------------------------
# STEP 4: Train/Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['Survived']),
    df['Survived'],
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# STEP 5: Define Transformers
# ------------------------------------------------------------
# 5.1 Handle Missing Values

# Missing value handler for mixed data types
trf1 = ColumnTransformer([
    
    # Numerical feature: Age (mean imputation)
    ('impute_age', SimpleImputer(), [2]),

    # calling index value of column for better performance

    
    # Categorical feature: Embarked (mode imputation)
    ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])
], 
remainder='passthrough')  # Maintain original features not being imputed

# 5.2 One-Hot Encode Categorical Features
# Categorical feature encoder for sex/embarked

trf2 = ColumnTransformer([
    
    # Convert categorical features to one-hot vectors
    ('ohe_sex_embarked', OneHotEncoder(
        sparse_output=False,            # Return dense array for better compatibility
        handle_unknown='ignore'  # Skip unknown categories in test data
    ), [1, 6])                   # Columns: Sex (1) and Embarked (6)
], 
remainder='passthrough')        # Preserve non-categorical features


# 5.3 Scale Numerical Features

trf3 = ColumnTransformer([
    # Normalize numerical features to 0-1 range
    ('scale', MinMaxScaler(), slice(0,10))  # Applies to columns 0-9 (end-exclusive)
])


# MinMaxScaler: Normalizes features to specified range (default)
# use MInMaxScaler for Feature Selection
# slice(0,10): Applies scaling to first 10 columns (indices 0-9)

# 5.4 Feature Selection
trf4 = SelectKBest(
    score_func=chi2,  # Statistical test for feature-target association
    k=8               # Number of top features to select
)

# 5.5 Classifier
trf5 = DecisionTreeClassifier()
# Decision Tree Classifier for final prediction


# ------------------------------------------------------------
# STEP 6: Create Pipeline
# ------------------------------------------------------------
# End-to-End Feature Processing Pipeline

pipe = Pipeline([
    # Stage 1: Data Imputation
    ('trf1', trf1),  # ColumnTransformer for age/embarked imputation
    
    # Stage 2: Categorical Encoding
    ('trf2', trf2),  # OHE for sex/embarked (handle_unknown='ignore')
    
    # Stage 3: Feature Scaling
    ('trf3', trf3),  # MinMaxScaler for first 10 features
    
    # Stage 4: Feature Selection
    ('trf4', trf4),  # SelectKBest with chi2 (k=8 features)
    
    # Stage 5: Final Model/Estimator
    ('trf5', trf5)   # Example: LogisticRegression() or other estimator
])

# ------------------------------------------------------------
# STEP 7: Train Model
# ------------------------------------------------------------
pipe.fit(X_train, y_train)

# ------------------------------------------------------------
# STEP 8: Explore Pipeline
# ------------------------------------------------------------
print("Pipeline Steps:", pipe.named_steps)
print("Imputer Statistics (Age):", pipe.named_steps['trf1'].transformers_[0][1].statistics_)
print("Imputer Statistics (Embarked):", pipe.named_steps['trf1'].transformers_[1][1].statistics_)

# ------------------------------------------------------------
# STEP 9: Prediction
# ------------------------------------------------------------
y_pred = pipe.predict(X_test)
print("Sample Predictions:", y_pred[:10])

# ------------------------------------------------------------
# STEP 10: Accuracy Score
# ------------------------------------------------------------
score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", score)

# ------------------------------------------------------------
# STEP 11: Cross Validation
# ------------------------------------------------------------
# cross validation using cross_val_score

cv_scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
).mean()
print("Cross-Validation Accuracy:", cv_scores)

# ------------------------------------------------------------
# STEP 12: Hyperparameter Tuning
# ------------------------------------------------------------
# gridsearchcv (use best fit value)
params = {
    'trf5__max_depth': [1,2,3,4,5,None]
}

grid = GridSearchCV(
    estimator=pipe,       # Full preprocessing + model pipeline
    param_grid=params,    # Dictionary of parameters to search
    cv=5,                 # 5-fold stratified validation
    scoring='accuracy',   # Primary evaluation metric
    n_jobs=-1,            # Use all CPU cores
    verbose=1             # Show progress messages
)

grid.fit(X_train, y_train)  # X_train should be raw unpreprocessed data
print("Best CV Score:", grid.best_score_)
print("Best Parameters:", grid.best_params_)

# ------------------------------------------------------------
# STEP 13: Export Pipeline
# ------------------------------------------------------------
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("Pipeline exported successfully as 'pipe.pkl'")

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates building an end-to-end ML pipeline on the Titanic dataset.")
print("Steps included: Imputation, Encoding, Scaling, Feature Selection, Model Training,")
print("Evaluation, Cross-Validation, Hyperparameter Tuning, and Export.")
print("\nKey Insights:")
print("1. Pipelines ensure reproducibility and prevent data leakage.")
print("2. They simplify deployment by combining preprocessing + model into one object.")
print("3. Hyperparameter tuning improves model performance systematically.")
print("\nRecruiter Takeaway: This script shows strong ML engineering skills,")
print("attention to detail, and professional communication — essential for production-ready ML work.")
print("============================================================")