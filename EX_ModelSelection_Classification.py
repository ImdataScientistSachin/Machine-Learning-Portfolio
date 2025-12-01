#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Executive Summary: Model Selection (Classification)
# ============================================================

# Purpose:
# This script demonstrates how to perform model selection for classification
# using multiple algorithms from scikit-learn. We compare models such as
# Logistic Regression, Decision Trees, Random Forests, Gradient Boosting,
# KNN, Naive Bayes, and Support Vector Machines on the Iris dataset.

# Why it matters:

# Recruiters and collaborators can see not only the code but also the
# reasoning behind each step. This makes the script a professional,
# tutorial-style report suitable for GitHub or LinkedIn.

# Dataset:
# Iris dataset — contains measurements of iris flowers (features) and species labels (targets).

# Workflow:
# 1. Illustrate basic Python logic with lists and tuples
# 2. Load and preprocess the Iris dataset
# 3. Train/test split
# 4. Fit multiple classification models
# 5. Evaluate performance
# 6. Summarize results

# ============================================================

# ==== Step 1: Illustrate Basic Python Logic ====

# Example with a list
X = [10, 20, 30, 40, 65, 78]
print("List values:")
for i in X:
    print(i)

# Example with a tuple
Tup = [('dhiraj', 36), ('Sangram', 25), ('Ashok', 19)]
print("\nTuple values:")
for Name, Age in Tup:
    print("Name:", Name, "| Age:", Age)

# ==== Step 2: Load Libraries ====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # clean, professional plot style

# ==== Step 3: Load Dataset ====

# The Iris dataset is a classic dataset for classification.
from sklearn.datasets import load_iris
Iris = load_iris()
print("\nIris dataset loaded successfully.")

# ==== Step 4: Transform Dataset ====

# Features (X) = measurements of iris flowers
# Target (Y) = species labels
X = Iris.data
Y = Iris.target
print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", Y.shape)

# ==== Step 5: Import Classification Models ====

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ==== Step 6: Train/Test Split ====

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)
print("\nTrain/Test split completed.")

# ==== Step 7: Prepare Models ====

# We create a list of models to test. Each model represents a different
# approach to classification, from linear to ensemble methods.
models = [
    ('LR', LogisticRegression()),                 # Logistic Regression
    ('DT', DecisionTreeClassifier()),             # Decision Tree
    ('RT', RandomForestClassifier(n_estimators=10)),  # Random Forest
    ('GB', GradientBoostingClassifier(n_estimators=10)),  # Gradient Boosting
    ('KNN', KNeighborsClassifier(n_neighbors=10)),       # KNN
    ('GN', GaussianNB()),                         # Naive Bayes
    ('SV', SVC(kernel='linear'))                  # Support Vector Machine
]
print("\nModels prepared for evaluation.")

# ==== Step 8: Fit and Evaluate Models ====

# Loop through each model, fit it to the training data, and evaluate
# performance on the test set. Scores represent accuracy.
print("\nModel Evaluation Results:")
for Name, Cls in models:
    model = Cls
    model.fit(X_train, y_train)  # train model
    score = model.score(X_test, y_test)  # evaluate model
    print(Name, ':', score)

# ==== Final Summary ====
# ------------------------------------------------------------
# Model Selection Results:
# - Multiple classification algorithms were tested on the Iris dataset.
# - Each model provides a different perspective:
#   * Logistic Regression: simple, interpretable baseline.
#   * Decision Tree: captures non-linear relationships.
#   * Random Forest & Gradient Boosting: ensemble methods, strong performance.
#   * KNN: similarity-based classification.
#   * Naive Bayes: probabilistic, fast baseline.
#   * SVM: effective for linear separation.
#
# Key Takeaway:
# This script demonstrates how to systematically compare classification models.
# The recruiter or collaborator can see both the technical workflow and
# the reasoning behind each choice, making it a polished, professional report.
# ------------------------------------------------------------