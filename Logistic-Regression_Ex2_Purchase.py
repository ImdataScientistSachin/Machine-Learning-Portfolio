
#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Logistic Regression** # to predict whether a user makes a purchase based on social network ads
# ================================================================
# This script demonstrates the application of **Logistic Regression**
# to predict whether a user makes a purchase based on social network ads.

# Purpose:
# - To classify outcomes (purchase vs. no purchase) using logistic regression.
# - To showcase the end-to-end workflow: data loading, preprocessing, model training,
#   prediction, and evaluation.

# Why it matters:
# - Logistic regression is a foundational classification algorithm widely used in
#   marketing, healthcare, and finance.
# - Recruiters value candidates who can explain not only how to implement models
#   but also why they are used, how to interpret results, and how to validate assumptions.

# Techniques highlighted:
# - Data handling with Pandas.
# - Train-test split for unbiased evaluation.
# - Logistic regression modeling with scikit-learn.
# - Model evaluation using confusion matrix and classification report.

# Statistical tests interpretation:
# - Confusion Matrix: Shows true positives, true negatives, false positives, false negatives.
# - Classification Report: Provides precision, recall, F1-score, and accuracy.

# ================================================================


# ==== Step 1: Import Libraries ====
# NumPy: numerical operations
# Pandas: data handling
# Seaborn/Matplotlib: visualization
# Recruiter-friendly note: Clean visuals and structured imports show professionalism.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')  # Consistent, professional plotting style


# ==== Step 2: Load Dataset ====
# Reading CSV file into a Pandas DataFrame.
# Pandas `read_csv` is a standard method for loading structured data.
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset  # Displays dataset for inspection


# ==== Step 3: Define Features and Target ====
# Independent variables (X): Age and Estimated Salary
# Dependent variable (Y): Purchased (binary outcome)
# `iloc` is used to select specific columns by index.
X = dataset.iloc[:, 2:4].values
X  # Inspect feature matrix

Y = dataset.iloc[:, 4].values
Y  # Inspect target vector


# ==== Step 4: Split Data into Training and Testing Sets ====
# Train-test split ensures unbiased evaluation.
# 80% training, 20% testing. `random_state=0` ensures reproducibility.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Inspect shapes to confirm split
X_train.shape  # Training features
X_test.shape   # Testing features
Y_train        # Training labels
Y_test.shape   # Testing labels


# ==== Step 5: Build Logistic Regression Model ====
# Logistic Regression is used for binary classification problems.
# It models the probability of an outcome using the logistic (sigmoid) function.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Train the model using training data
model.fit(X_train, Y_train)


# ==== Step 6: Make Predictions ====
# Use the trained model to predict outcomes on test data.
Y_pred = model.predict(X_test)
Y_pred  # Predicted labels


# ==== Step 7: Evaluate Model Performance ====
# Confusion Matrix: Evaluates classification performance by comparing predicted vs actual.
# Classification Report: Provides precision, recall, F1-score, and accuracy.
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix interpretation:
# - Top-left: True Negatives (correctly predicted no purchase)
# - Top-right: False Positives (incorrectly predicted purchase)
# - Bottom-left: False Negatives (missed purchase)
# - Bottom-right: True Positives (correctly predicted purchase)
print(confusion_matrix(Y_test, Y_pred))

# Classification Report interpretation:
# - Precision: Of predicted positives, how many are correct.
# - Recall: Of actual positives, how many were identified.
# - F1-score: Harmonic mean of precision and recall.
# - Accuracy: Overall correctness of the model.
print(classification_report(Y_test, Y_pred))


# ================================================================
# FINAL SUMMARY
# ================================================================
# - We successfully applied logistic regression to classify purchase behavior.
# - The workflow included data loading, preprocessing, model training, prediction,
#   and evaluation.
# - Confusion matrix and classification report provided insights into model performance:
#   * High accuracy indicates the model generalizes well.
#   * Precision and recall highlight trade-offs in classification.
# - Logistic regression proved effective for binary classification tasks.

# ================================================================