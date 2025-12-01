#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Classification (Supervised - Learning ) Using sklearn
# ================================================================

# This script demonstrates Logistic Regression using scikit-learn.
#
# Purpose:
# - Logistic Regression is a supervised learning algorithm used for binary classification.
# - Here, we predict whether a student is admitted based on SAT scores.
# formula : px = e(b0 + b1*x1 + b2*x2+ bn*xn) / 1 + e(b0 + b1*x1 + b2*x2+ bn*xn


# Why It Matters (Recruiter-Friendly Narrative):
# - Logistic regression is a foundational classification technique in machine learning.
# - It is widely applied in admissions, credit scoring, medical diagnosis, and spam detection.
# - Demonstrating mastery of logistic regression shows ability to handle categorical outcomes,
#   evaluate models, and communicate results clearly.

# Key Concepts:
# - Binary Outcome: Dependent variable takes values {0,1}.
# - Logistic Function (Sigmoid): Maps real values into (0,1), representing probability.
# - Confusion Matrix: Evaluates classification performance by comparing predictions vs. actuals.
# - Precision, Recall, F1-Score: Key metrics for model evaluation.

# # confusion metrics & classification report

#  Confusion matrix and classification report are essential tools for evaluating the performance of a classification model.

#  True Negatives (TN): The number of correct negative predictions.
# 
#  False Positives (FP): The number of incorrect positive predictions (Type I error).
# 
#  False Negatives (FN): The number of incorrect negative predictions (Type II error).
# 
#  True Positives (TP): The number of correct positive predictions.
# ================================================================


# ==== Step 1: Import Required Libraries ====
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')  # Professional plot styling


# ==== Step 2: Load Dataset ====

# Dataset contains SAT scores and admission outcomes (Yes/No).
dataset = pd.read_csv('admitance.csv')
dataset  # Display raw dataset


# ==== Step 3: Convert Categorical to Numerical ====
# Logistic regression requires numerical dependent variables.
# Mapping: Yes → 1, No → 0
dataset['Admitted'] = dataset['Admitted'].map({'Yes': 1, 'No': 0})
dataset  # Display transformed dataset


# ==== Step 4: Define Predictor (X) and Target (Y) ====

# X = SAT scores (independent variable)
# Y = Admission outcome (dependent variable)
X = dataset.iloc[:, [0]].values
Y = dataset.iloc[:, 1].values


# ==== Step 5: Train-Test Split ====

# Splitting dataset into training (80%) and testing (20%) sets.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# ==== Step 6: Build Logistic Regression Model ====
from sklearn.linear_model import LogisticRegression

# Initialize and train model
model = LogisticRegression()
model.fit(X_train, Y_train)


# ==== Step 7: Model Predictions ====

# Predict admission outcomes for test set
y_pred = model.predict(X_test)
y_pred  # Display predictions


# ==== Step 8: Evaluate Model Performance ====

# Confusion matrix and classification report are essential tools for evaluation.
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
#  The ConfusionMatrixDisplay class provides a convenient way to display the confusion matrix as a heatmap, which can help you quickly understand the performance of your classification model.

cm = confusion_matrix(Y_test, y_pred)
print(cm)

# Interpretation:
# TN = 14 → Correctly predicted "Not Admitted"
# FP = 1  → Incorrectly predicted "Admitted" (Type I error)
# FN = 1  → Incorrectly predicted "Not Admitted" (Type II error)
# TP = 18 → Correctly predicted "Admitted"

# Classification Report
print(classification_report(Y_test, y_pred))

# Explanation of Metrics:
# - Precision: Correct positive predictions / Total predicted positives
# - Recall: Correct positive predictions / Total actual positives
# - F1-Score: Harmonic mean of Precision and Recall
# - Support: Number of actual occurrences of each class


# ==== Step 9: Visualize Confusion Matrix ====

from sklearn.metrics import ConfusionMatrixDisplay


# Display confusion matrix as heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.grid(False)
plt.show()


# ================================================================
# FINAL SUMMARY
# ================================================================

# Key Takeaways:
# - Logistic regression successfully models binary outcomes (admitted vs. not admitted).
# - SAT scores positively influence admission probability.
# - Confusion matrix and classification report provide detailed evaluation of model performance.
#
# Professional Impact:
# - Demonstrates ability to preprocess categorical data, fit logistic models,
#   and evaluate classification results using industry-standard metrics.
# - Shows statistical literacy and ability to communicate results clearly.
# - Portfolio-ready project for GitHub/LinkedIn showcasing supervised learning skills.

#  Precision: The ratio of correctly predicted positive observations to the total predicted positives. FOR ( + )=  TP / TP + FN , FOR ( - )=  TN / TN + FN
# #### Recall: The ratio of correctly predicted positive observations to all observations in the actual class. FOR ( + )=  TP / TP + FP , FOR ( - )=  TN / TN + FN

# F1-Score: The weighted average of Precision and Recall. 2 (Precision*Recall) / (Precision * Recall)
# Support ( Accursncy ): The number of actual occurrences of the class in the dataset.  

# ================================================================