#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Handling Missing Numerical Data - Mean & Median Imputation
#
# Definition:
# Simple imputation replaces missing values with statistical measures
# such as mean or median. Median is robust to outliers, while mean
# preserves overall average.
#
# Objective:
# Demonstrate how mean/median imputation affects variance, distribution,
# and correlation in numerical features.
#
# Why Important:
# - Prevents dropping rows with missing values.
# - Maintains dataset size for modeling.
# - But can distort variance and relationships if not chosen carefully.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Load Libraries
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv('titanic_toy.csv')
print("Initial dataset sample:\n", df.head())

# ------------------------------------------------------------
# STEP 3: Explore Dataset
# ------------------------------------------------------------
df.info()
print("Fraction of missing values per column:\n", df.isnull().mean())
# INTERPRETATION: Age and Fare contain missing values. These will be imputed.

# ------------------------------------------------------------
# STEP 4: Prepare Features and Target
# ------------------------------------------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

print("Train/Test shapes:", X_train.shape, X_test.shape)
print("Missingness in training set:\n", X_train.isnull().mean())

# ------------------------------------------------------------
# STEP 5: Compute Mean & Median
# ------------------------------------------------------------
# find the mean and median od Age and Fare column and store in new column

mean_age = X_train['Age'].mean()
median_age = X_train['Age'].median()
mean_fare = X_train['Fare'].mean()
median_fare = X_train['Fare'].median()

print("Age mean:", mean_age, "Age median:", median_age)
print("Fare mean:", mean_fare, "Fare median:", median_fare)

# ------------------------------------------------------------
# STEP 6: Manual Imputation
# ------------------------------------------------------------
# fill null values with mean and median values

X_train['Age_median'] = X_train['Age'].fillna(median_age)
X_train['Age_mean'] = X_train['Age'].fillna(mean_age)
X_train['Fare_median'] = X_train['Fare'].fillna(median_fare)
X_train['Fare_mean'] = X_train['Fare'].fillna(mean_fare)

print("Sample after imputation:\n", X_train.sample(5))
# show original column value and modified new values

# ------------------------------------------------------------
# STEP 7: Variance Comparison
# ------------------------------------------------------------
print('Original Age variance: ', X_train['Age'].var())
print('Age variance after median imputation: ', X_train['Age_median'].var())
print('Age variance after mean imputation: ', X_train['Age_mean'].var())
print()
print('Original Fare variance: ', X_train['Fare'].var())
print('Fare variance after median imputation: ', X_train['Fare_median'].var())
print('Fare variance after mean imputation: ', X_train['Fare_mean'].var())
# INTERPRETATION: Median imputation preserves variance better when outliers exist.
# Mean imputation can shrink variance if missingness is large.

# ------------------------------------------------------------
# STEP 8: Distribution Plots
# ------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
X_train['Age'].plot(kind='kde', ax=ax, label='Original Age')
X_train['Age_median'].plot(kind='kde', ax=ax, color='red', label='Age Median')
X_train['Age_mean'].plot(kind='kde', ax=ax, color='green', label='Age Mean')
ax.legend(loc='best')
plt.title("Age Distribution: Original vs Imputed")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['Fare'].plot(kind='kde', ax=ax, label='Original Fare')
X_train['Fare_median'].plot(kind='kde', ax=ax, color='red', label='Fare Median')
X_train['Fare_mean'].plot(kind='kde', ax=ax, color='green', label='Fare Mean')
ax.legend(loc='best')
plt.title("Fare Distribution: Original vs Imputed")
plt.show()
# INTERPRETATION: Median imputation aligns closely with original distribution,
# while mean imputation shifts the curve slightly.

# ------------------------------------------------------------
# STEP 9: Covariance & Correlation
# ------------------------------------------------------------
print("Covariance Matrix:\n", X_train.cov())
print("Correlation Matrix:\n", X_train.corr())
# INTERPRETATION: Imputation alters correlations slightly,
# but median tends to preserve relationships better.

# ------------------------------------------------------------
# STEP 10: Boxplots
# ------------------------------------------------------------
X_train[['Age','Age_median','Age_mean']].boxplot()
plt.title("Boxplot: Age Original vs Imputed")
plt.show()

X_train[['Fare','Fare_median','Fare_mean']].boxplot()
plt.title("Boxplot: Fare Original vs Imputed")
plt.show()

# ------------------------------------------------------------
# STEP 11: Using Sklearn SimpleImputer
# ------------------------------------------------------------
imputer1 = SimpleImputer(strategy='median')
imputer2 = SimpleImputer(strategy='mean')

trf = ColumnTransformer([
    ('imputer1', imputer1, ['Age']),
    ('imputer2', imputer2, ['Fare'])
], remainder='passthrough')

trf.fit(X_train)

print("Median imputer statistic (Age):", trf.named_transformers_['imputer1'].statistics_)
print("Mean imputer statistic (Fare):", trf.named_transformers_['imputer2'].statistics_)

X_train_transformed = trf.transform(X_train)
X_test_transformed = trf.transform(X_test)

print("Transformed training sample:\n", X_train_transformed[:5])

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates Mean & Median Imputation for numerical features.")
print("Key Insights:")
print("1. Median imputation is robust to outliers and preserves variance better.")
print("2. Mean imputation maintains averages but can distort distributions.")
print("3. Both methods prevent row loss, but choice depends on data distribution.")
print("4. Sklearn’s SimpleImputer automates imputation for reproducibility.")
print("\nRecruiter Takeaway: This script shows practical preprocessing skills,")
print("awareness of statistical impacts, and clear communication — essential traits for applied ML roles.")
print("============================================================")