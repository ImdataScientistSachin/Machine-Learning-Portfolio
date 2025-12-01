#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary : AUC & ROC Introduction
# ================================================================

# This script explains and demonstrates **AUC-ROC (Area Under the Curve - Receiver Operating Characteristic)**,
# a key metric for evaluating binary classification models.

# The Area Under the Curve - Receiver Operating Characteristic (AUC-ROC) is a crucial metric in evaluating the performance of classification models, particularly in binary classification tasks. It provides a single scalar value that summarizes the model's ability to distinguish between positive and negative classes across various threshold settings.

# Why AUC-ROC matters:
# - Provides a single scalar value summarizing model discrimination ability.
# - ROC curve visualizes trade-off between True Positive Rate (TPR) and False Positive Rate (FPR).
# - AUC quantifies overall performance: higher AUC → better separation of classes.
#
# Recruiter-Friendly Takeaway:
# This script shows practical application of **AUC-ROC evaluation** with Logistic Regression,
# highlighting awareness of statistical metrics, visualization, and interpretability.

# ##### AUC Interpretation : 
#  The AUC quantifies the overall ability of the model to discriminate between positive and negative classes. Its value ranges from 0 to 1:
#  AUC = 1: Perfect model.
# AUC = 0.5: No discrimination capability (equivalent to random guessing).
# AUC < 0.5: Indicates a model that performs worse than random chance.
# ================================================================


# ==== Step 1: Import Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score


# ==== Step 2: Generate Synthetic Dataset ====
# make_classification generates a binary dataset with 1000 samples and 4 features.
X, Y = make_classification(n_samples=1000, n_classes=2, n_features=4, random_state=0)

#  make_classification: This is a function that generates random data for classification tasks.
#  n_samples=1000: It means we want to create 1,000 examples (or data points or Rows).
#  n_classes=2: This indicates that there will be 2 categories (or classes) to classify the data into, making it a binary classification problem.
#  n_features=4: Each example will have 4 different characteristics (or features).
#  random_state=0: This is like a seed that helps us get the same random data every time we run the code. It ensures consistency.


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# ==== Step 3: Train Logistic Regression Model ====
model1 = LogisticRegression()
model1.fit(X_train, y_train)

# Predictions
Predic1 = model1.predict(X_test)

# Probabilities (needed for ROC/AUC)
Pred_prob1 = model1.predict_proba(X_test).round(3)


# ==== Step 4: Confusion Matrix & Classification Report ====
print("Confusion Matrix:\n", confusion_matrix(y_test, Predic1))
print("\nClassification Report:\n", classification_report(y_test, Predic1))


# ==== Step 5: ROC Curve Values ====
Fpr1, Tpr1, Threshold1 = roc_curve(y_test, Pred_prob1[:,1], pos_label=1)

print("False Positive Rates:", Fpr1)
print("True Positive Rates:", Tpr1)
print("Thresholds:", Threshold1.round(3))


# ==== Step 6: AUC Score ====
AUC_score = roc_auc_score(y_test, Pred_prob1[:,1])
print("AUC Score:", AUC_score)

# Interpretation:
# - AUC = 1 → Perfect model
# - AUC = 0.5 → Random guessing
# - AUC < 0.5 → Worse than random


# ==== Step 7: Plot ROC Curve ====
plt.plot(Fpr1, Tpr1, marker='o', ls='--', label=f'Logistic Regression (AUC={AUC_score:.2f})')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc=0)
plt.show()


# ================================================================
# 📊 Statistical & Conceptual Notes
# ================================================================
# - True Positive Rate (TPR / Recall): TP / (TP + FN)
# - False Positive Rate (FPR): FP / (FP + TN)
# - ROC Curve: Plots TPR vs FPR across thresholds.
# - AUC: Measures overall discrimination ability.
#
# Awareness of these metrics demonstrates readiness for professional-level model evaluation.


# ================================================================
# 📌 Final Summary
# ================================================================
# - Logistic Regression was trained on synthetic binary classification data.
# - ROC curve visualized trade-off between sensitivity and specificity.
# - AUC score quantified model performance (closer to 1 → better).
# - This highlights importance of evaluating models beyond accuracy.

# ================================================================