#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary : AUC & ROC On Dataset

# ================================================================
# This script compares **Logistic Regression** and **K-Nearest Neighbors (KNN)** classifiers
# using the **Social Network Ads dataset**. The evaluation focuses on:
# - Confusion Matrix & Classification Report (precision, recall, F1-score).
# - ROC (Receiver Operating Characteristic) curves.
# - AUC (Area Under Curve) scores.

# Why this matters:
# - ROC/AUC provides a robust measure of model performance beyond accuracy.
# - Logistic Regression is a probabilistic linear model, while KNN is a non-parametric model.
# - Comparing them highlights trade-offs in interpretability, complexity, and predictive power.

# Recruiter-Friendly Takeaway:
# This script demonstrates proficiency in **model evaluation techniques** and ability to
# communicate results clearly with visualizations and statistical metrics.
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


# ==== Step 2: Load Dataset ====
Dataset = pd.read_csv('Social_Network_Ads.csv')
Dataset.head()

# Select features (Age, Estimated Salary) and target (Purchased)
X = Dataset.iloc[:, [2, 3]].values
Y = Dataset.iloc[:, -1].values


# ==== Step 3: Train-Test Split ====
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# ==== Step 4: Initialize Models ====
model1 = LogisticRegression()
model2 = KNeighborsClassifier(n_neighbors=15)


# ==== Step 5: Train Models ====
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)


# ==== Step 6: Predictions ====
Pred_1 = model1.predict(X_test)
Pred_2 = model2.predict(X_test)


# ==== Step 7: Probabilities ====
Prob1 = model1.predict_proba(X_test)
Prob2 = model2.predict_proba(X_test)


# ==== Step 8: Confusion Matrix & Classification Report ====
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, Pred_1))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, Pred_2))

print("\nLogistic Regression Report:\n", classification_report(y_test, Pred_1))
print("KNN Report:\n", classification_report(y_test, Pred_2))


# ==== Step 9: ROC Curve Values ====
FPR1, TPR1, Threshold1 = roc_curve(y_test, Prob1[:,1], pos_label=1)
FPR2, TPR2, Threshold2 = roc_curve(y_test, Prob2[:,1], pos_label=1)


# ==== Step 10: AUC Scores ====
AUC_score1 = roc_auc_score(y_test, Prob1[:,1])
AUC_score2 = roc_auc_score(y_test, Prob2[:,1])

print("Logistic Regression AUC:", AUC_score1)
print("KNN AUC:", AUC_score2)


# ==== Step 11: Plot ROC Curves ====
plt.plot(FPR1, TPR1, marker='o', ls='--', label=f'Logistic (AUC={AUC_score1:.2f})')
plt.plot(FPR2, TPR2, marker='o', ls='--', label=f'KNN (AUC={AUC_score2:.2f})')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Comparison (Social Network Ads)')
plt.legend(loc=0)
plt.show()


# ================================================================
# 📊 Statistical & Conceptual Notes
# ================================================================
# - Confusion Matrix: Shows TP, TN, FP, FN counts.
# - Classification Report: Precision, Recall, F1-score for each class.
# - ROC Curve: Plots TPR vs FPR across thresholds.
# - AUC: Measures overall ability to distinguish between classes.

# Interpretation:
# - Logistic Regression provides smoother ROC curve due to probabilistic modeling.
# - KNN performance depends heavily on choice of neighbors (n_neighbors=15 here).
# - Higher AUC indicates better discrimination ability.
#
# Awareness of these metrics demonstrates readiness for professional-level model evaluation.


# ================================================================
# 📌 Final Summary
# ================================================================

# - Logistic Regression and KNN were trained on the Social Network Ads dataset.
# - ROC/AUC analysis provided a robust comparison of their performance.
# - Logistic Regression achieved AUC ≈ {AUC_score1:.2f}, while KNN achieved AUC ≈ {AUC_score2:.2f}.
# - This highlights the importance of evaluating models beyond accuracy, using ROC/AUC.
#

# ================================================================