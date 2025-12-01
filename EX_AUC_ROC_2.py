#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary : AUC ROC
# ================================================================

# This script compares **two classification models** — Logistic Regression and K-Nearest Neighbors (KNN) —
# using **AUC (Area Under Curve)** and **ROC (Receiver Operating Characteristic)** analysis.


# Why this matters:
# - ROC curves visualize the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR).
# - AUC quantifies model performance: higher AUC → better discrimination between classes.
# - Comparing models with ROC/AUC provides recruiter-friendly evidence of model evaluation skills.

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score


# ==== Step 2: Generate Synthetic Dataset ====
# make_classification creates a binary classification dataset with 1000 samples and 4 features.
X, Y = make_classification(n_samples=1000, n_classes=2, n_features=4, random_state=0)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# ==== Step 3: Initialize Models ====
model1 = LogisticRegression()
model2 = KNeighborsClassifier(n_neighbors=15)


# ==== Step 4: Train Models ====
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)


# ==== Step 5: Predictions ====
Pred_1 = model1.predict(X_test)
Pred_2 = model2.predict(X_test)


# ==== Step 6: Probabilities ====
# Probabilities are required for ROC/AUC analysis.
Prob1 = model1.predict_proba(X_test)
Prob2 = model2.predict_proba(X_test)


# ==== Step 7: Confusion Matrix & Classification Report ====
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, Pred_1))
print("Logistic Regression Report:\n", classification_report(y_test, Pred_1))


# ==== Step 8: ROC Curve Values ====
FPR1, TPR1, Threshold1 = roc_curve(y_test, Prob1[:,1], pos_label=1)
FPR2, TPR2, Threshold2 = roc_curve(y_test, Prob2[:,1], pos_label=1)


# ==== Step 9: AUC Scores ====
AUC_score1 = roc_auc_score(y_test, Prob1[:,1])
AUC_score2 = roc_auc_score(y_test, Prob2[:,1])

print("Logistic Regression AUC:", AUC_score1)
print("KNN AUC:", AUC_score2)


# ==== Step 10: Plot ROC Curves ====
plt.plot(FPR1, TPR1, marker='o', ls='--', label=f'Logistic (AUC={AUC_score1:.2f})')
plt.plot(FPR2, TPR2, marker='o', ls='--', label=f'KNN (AUC={AUC_score2:.2f})')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Comparison')
plt.legend(loc=0)
plt.show()


# ================================================================
# 📊 Statistical & Conceptual Notes
# ================================================================
# - Confusion Matrix: Shows counts of TP, TN, FP, FN.
# - Classification Report: Includes precision, recall, F1-score.
# - ROC Curve: Plots TPR vs FPR at different thresholds.
# - AUC: Measures overall ability to distinguish between classes.
#
# Interpretation:
# - Logistic Regression typically produces smoother ROC curves due to probabilistic modeling.
# - KNN performance depends on choice of neighbors (n_neighbors=15 here).
# - Higher AUC → better model discrimination.
#
# Awareness of these metrics demonstrates professional-level model evaluation skills.


# ================================================================
# 📌 Final Summary
# ================================================================
# - Logistic Regression and KNN were trained on the same dataset.
# - ROC/AUC analysis provided a robust comparison of their performance.
# - Logistic Regression achieved AUC ≈ {AUC_score1:.2f}, while KNN achieved AUC ≈ {AUC_score2:.2f}.
# - This highlights the importance of evaluating models beyond accuracy, using ROC/AUC.
#

# ================================================================