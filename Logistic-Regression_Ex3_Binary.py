#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Logistic Regression binary  classification
# ================================================================

# This script demonstrates the use of **Logistic Regression** for a binary  classification problem: predicting admission outcomes based on features  such as exam scores and gender.

# Purpose:
# - To classify categorical outcomes (Admitted: Yes/No) using logistic regression.
# - To showcase the workflow: data preprocessing, model training, prediction,
#   and evaluation.

# Why it matters:
# - Logistic regression is a fundamental classification technique widely used
#   in education, healthcare, and marketing.
# - Recruiters value candidates who can explain not only how to implement models
#   but also why they are used, how to interpret results, and how to validate assumptions.

# Techniques highlighted:
# - Data preprocessing: converting categorical values to numerical.
# - Train-test split for unbiased evaluation.
# - Logistic regression modeling with scikit-learn.
# - Model evaluation using confusion matrix and classification report.

# Statistical tests interpretation:
# - Confusion Matrix: Shows true positives, true negatives, false positives, false negatives.
# - Classification Report: Provides precision, recall, F1-score, and accuracy.

# This script is written as a self-documenting tutorial/report, making it portfolio-ready
# for GitHub or LinkedIn.
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
dataset = pd.read_csv('binary_log.csv')
dataset  # Displays dataset for inspection


# ==== Step 3: Convert Categorical Values to Numerical ====
# Logistic regression requires numerical inputs.
# Mapping categorical values (Yes/No, Male/Female) to binary numbers.
dataset['Admitted'] = dataset['Admitted'].map({'Yes': 1, 'No': 0})
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
dataset  # Inspect transformed dataset


# ==== Step 4: Define Features and Target ====
# Dependent variable (Y): Admitted (binary outcome)
# Independent variables (X): Exam Score and Gender
# `iloc` is used to select specific columns by index.
Y = dataset.iloc[:, 1].values
Y  # Inspect target vector

X = dataset.iloc[:, [0, 2]].values
X  # Inspect feature matrix


# ==== Step 5: Split Data into Training and Testing Sets ====
# Train-test split ensures unbiased evaluation.
# 80% training, 20% testing. `random_state=0` ensures reproducibility.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Inspect shapes to confirm split
X_train.shape  # Training features
Y_train.shape  # Training labels
X_test.shape   # Testing features
Y_test.shape   # Testing labels


# ==== Step 6: Build Logistic Regression Model ====
# Logistic Regression is used for binary classification problems.
# It models the probability of an outcome using the logistic (sigmoid) function.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Train the model using training data
model.fit(X_train, Y_train)


# ==== Step 7: Make Predictions ====
# Use the trained model to predict outcomes on test data.
Y_pred = model.predict(X_test)
Y_pred  # Predicted labels


# ==== Step 8: Evaluate Model Performance ====
# Confusion Matrix: Evaluates classification performance by comparing predicted vs actual.
# Classification Report: Provides precision, recall, F1-score, and accuracy.
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix interpretation:
# - Top-left: True Negatives (correctly predicted not admitted)
# - Top-right: False Positives (incorrectly predicted admitted)
# - Bottom-left: False Negatives (missed admitted cases)
# - Bottom-right: True Positives (correctly predicted admitted)
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

# - We successfully applied logistic regression to classify admission outcomes.
# - The workflow included data preprocessing, model training, prediction,
#   and evaluation.
# - Confusion matrix and classification report provided insights into model performance:
#   * High accuracy indicates the model generalizes well.
#   * Precision and recall highlight trade-offs in classification.
# - Logistic regression proved effective for binary classification tasks.
#

# ================================================================