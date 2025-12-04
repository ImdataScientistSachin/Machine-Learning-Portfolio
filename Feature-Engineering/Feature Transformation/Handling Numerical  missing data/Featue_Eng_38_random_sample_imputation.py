#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Handle Missing Values - Random Sample Imputation

# #### Random sample imputation replaces missing values with randomly selected observed values from the same variable, preserving the original data distribution while handling missingness. This method is applicable to both numerical and categorical data 


# Objective:
# Replace missing values with randomly selected observed values
# from the same variable. This preserves the original distribution
# while handling missingness.
#
# Why Important:
# - Mean/median imputation distorts variance and distribution.
# - Random sampling maintains natural variability.
# - Works for both numerical and categorical features.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Import Libraries
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv('train.csv', usecols=['Age','Fare','Survived'])
print("Initial dataset sample:\n", df.head())

# ------------------------------------------------------------
# STEP 3: Explore Missingness
# ------------------------------------------------------------
missing_percentages = df.isnull().mean() * 100
print("Percentage of missing values per column:\n", missing_percentages)
# INTERPRETATION: Age has missing values; Fare is mostly complete.
# Knowing missingness guides imputation strategy.

# ------------------------------------------------------------
# STEP 4: Prepare Features and Target
# ------------------------------------------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

print("Training sample:\n", X_train.head())

# ------------------------------------------------------------
# STEP 5: Duplicate Column for Imputation
# ------------------------------------------------------------
# WHY: Preserve original Age column for comparison.
# WHAT: Create 'Age_imputed' column.
# HOW: Copy Age into new column.
X_train['Age_imputed'] = X_train['Age']
X_test['Age_imputed'] = X_test['Age']

# ------------------------------------------------------------
# STEP 6: Random Sample Imputation (Numerical)
# ------------------------------------------------------------
# WHY: Fill missing Age values with random samples from observed Age.
# WHAT: Sample without replacement to preserve distribution.
# HOW: Use .sample() from non-missing values.
X_train.loc[X_train['Age_imputed'].isnull(), 'Age_imputed'] = (
    X_train['Age'].dropna().sample(X_train['Age'].isnull().sum()).values
)
X_test.loc[X_test['Age_imputed'].isnull(), 'Age_imputed'] = (
    X_train['Age'].dropna().sample(X_test['Age'].isnull().sum()).values
)

# ------------------------------------------------------------
# STEP 7: Distribution Comparison
# ------------------------------------------------------------
sns.kdeplot(X_train['Age'], label='Original')
sns.kdeplot(X_train['Age_imputed'], label='Imputed')
plt.legend()
plt.title("Age Distribution: Original vs Random Imputation")
plt.show()
# INTERPRETATION: Imputed distribution closely follows original,
# unlike mean/median which would flatten variance.

# ------------------------------------------------------------
# STEP 8: Variance & Covariance Analysis
# ------------------------------------------------------------
print('Original Age variance: ', X_train['Age'].var())
print('Variance after random imputation: ', X_train['Age_imputed'].var())
# INTERPRETATION: Variance is preserved, showing random imputation
# maintains natural variability.

print("Covariance matrix:\n", X_train[['Fare','Age','Age_imputed']].cov())
# INTERPRETATION: Covariance between Age and Fare remains stable,
# meaning relationships are not distorted.

# ------------------------------------------------------------
# STEP 9: Boxplot Comparison
# ------------------------------------------------------------
X_train[['Age','Age_imputed']].boxplot()
plt.title("Boxplot: Original vs Imputed Age")
plt.show()

# ------------------------------------------------------------
# STEP 10: Deterministic Sampling Example
# ------------------------------------------------------------
# WHY: Use Fare as random seed for reproducibility.
# WHAT: Sample one Age value per test row.
# HOW: int(Fare) as seed ensures deterministic sampling.
for idx, observation in X_test.iterrows():
    sampled_value = X_train['Age'].dropna().sample(1, random_state=int(observation['Fare']))
    print(f"Sampled value for test row {idx}: {sampled_value.values}")

# ------------------------------------------------------------
# STEP 11: Random Sample Imputation (Categorical Example)
# ------------------------------------------------------------
data = pd.read_csv('house-train.csv', usecols=['GarageQual','FireplaceQu','SalePrice'])
print("House dataset sample:\n", data.head())

print("Missing percentages:\n", data.isnull().mean() * 100)

X = data
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Duplicate categorical columns for imputation
# Create a new column 'GarageQual_imputed' in both training and test sets,
# initially copying the original 'GarageQual' values.
# This allows you to perform imputation on 'GarageQual_imputed' without altering the original data.


X_train['GarageQual_imputed'] = X_train['GarageQual']
X_test['GarageQual_imputed'] = X_test['GarageQual']
X_train['FireplaceQu_imputed'] = X_train['FireplaceQu']
X_test['FireplaceQu_imputed'] = X_test['FireplaceQu']

# Randomly impute GarageQual
X_train.loc[X_train['GarageQual_imputed'].isnull(), 'GarageQual_imputed'] = (
    X_train['GarageQual'].dropna().sample(X_train['GarageQual'].isnull().sum()).values
)
X_test.loc[X_test['GarageQual_imputed'].isnull(), 'GarageQual_imputed'] = (
    X_train['GarageQual'].dropna().sample(X_test['GarageQual'].isnull().sum()).values
)

# Randomly impute FireplaceQu
X_train.loc[X_train['FireplaceQu_imputed'].isnull(), 'FireplaceQu_imputed'] = (
    X_train['FireplaceQu'].dropna().sample(X_train['FireplaceQu'].isnull().sum()).values
)
X_test.loc[X_test['FireplaceQu_imputed'].isnull(), 'FireplaceQu_imputed'] = (
    X_train['FireplaceQu'].dropna().sample(X_test['FireplaceQu'].isnull().sum()).values
)

# ------------------------------------------------------------
# STEP 12: Distribution Comparison (Categorical)
# ------------------------------------------------------------

# Compare the distribution of 'GarageQual' categories before and after imputation.
# The first column shows the relative frequency of each category in the original data (excluding missing values).
# The second column shows the relative frequency after imputation (no missing values).

temp = pd.concat([
    X_train['GarageQual'].value_counts() / len(X_train['GarageQual'].dropna()),
    X_train['GarageQual_imputed'].value_counts() / len(X_train)
], axis=1)
temp.columns = ['original','imputed']
print("GarageQual distribution comparison:\n", temp)


# Compare the distribution of 'FireplaceQu' categories before and after imputation.
# 'original' shows the relative frequency in the original data (excluding missing values).
# 'imputed' shows the frequency after imputation, using the total number of rows in df.
temp = pd.concat([
    X_train['FireplaceQu'].value_counts() / len(X_train['FireplaceQu'].dropna()),
    X_train['FireplaceQu_imputed'].value_counts() / len(data)
], axis=1)
temp.columns = ['original','imputed']
print("FireplaceQu distribution comparison:\n", temp)

# ------------------------------------------------------------
# STEP 13: KDE Plots by Category
# ------------------------------------------------------------
for category in X_train['FireplaceQu'].dropna().unique():
    sns.kdeplot(
        X_train.loc[X_train['FireplaceQu'] == category, 'SalePrice'],
        label=category
    )
plt.xlabel('SalePrice')
plt.ylabel('Density')
plt.title('SalePrice Distribution by Fireplace Quality')
plt.legend(title='FireplaceQu')
plt.show()

for category in X_train['FireplaceQu_imputed'].dropna().unique():
    sns.kdeplot(
        X_train.loc[X_train['FireplaceQu'] == category, 'SalePrice'],
        label=category
    )
plt.xlabel('SalePrice')
plt.ylabel('Density')
plt.title('SalePrice Distribution by Fireplace Quality (Imputed)')
plt.legend(title='FireplaceQu_Imputation')
plt.show()

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates Random Sample Imputation for both numerical and categorical features.")
print("Key Insights:")
print("1. Random sampling preserves variance and distribution, unlike mean/median imputation.")
print("2. It avoids biasing models by maintaining natural variability.")
print("3. For categorical features, random imputation maintains category proportions.")
print("\nRecruiter Takeaway: This script shows advanced feature engineering skills,")
print("awareness of statistical impacts, and clear communication — essential traits for applied ML roles.")
print("============================================================")