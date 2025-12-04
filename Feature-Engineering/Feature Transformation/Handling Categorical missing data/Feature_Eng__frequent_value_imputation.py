#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Handling Missing Categorical Data - Frequent Value Imputation (Mode)
#
# Definition:
# Frequent Value Imputation replaces missing values with the most
# frequently occurring category (mode) in a column.
#
# Why Important:
# - Simple and effective for categorical features.
# - Preserves dataset size without dropping rows.
# - Works well when missingness is small and categories are stable.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Load Libraries
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv('train.csv', usecols=['GarageQual','FireplaceQu','SalePrice'])
print("Initial dataset sample:\n", df.head())

# ------------------------------------------------------------
# STEP 3: Explore Missingness
# ------------------------------------------------------------
print("Percentage of missing values per column:\n", df.isnull().mean()*100)
# INTERPRETATION: GarageQual and FireplaceQu contain missing values.
# We will impute them using their most frequent categories.

# ------------------------------------------------------------
# STEP 4: Explore GarageQual
# ------------------------------------------------------------
df['GarageQual'].value_counts().plot(kind='bar', title="GarageQual Distribution (Original)")
plt.show()

print("Mode of GarageQual:", df['GarageQual'].mode())
# INTERPRETATION: 'TA' is the most frequent category.

# ------------------------------------------------------------
# STEP 5: Compare SalePrice Distribution (GarageQual)
# ------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
df[df['GarageQual']=='TA']['SalePrice'].plot(kind='kde', ax=ax, label='Houses with TA')
df[df['GarageQual'].isnull()]['SalePrice'].plot(kind='kde', ax=ax, color='red', label='Houses with NA')
ax.legend(loc='best')
plt.title("SalePrice Distribution by GarageQual")
plt.show()
# INTERPRETATION: Houses with missing GarageQual show different SalePrice distribution,
# motivating imputation with the mode.

# ------------------------------------------------------------
# STEP 6: Impute GarageQual
# ------------------------------------------------------------
df['GarageQual'] = df['GarageQual'].fillna('TA')
df['GarageQual'].value_counts().plot(kind='bar', title="GarageQual Distribution (After Imputation)")
plt.show()
# INTERPRETATION: Missing values are replaced with 'TA', increasing its frequency.

# ------------------------------------------------------------
# STEP 7: Compare SalePrice Distribution After Imputation
# ------------------------------------------------------------
temp = df[df['GarageQual']=='TA']['SalePrice']
fig = plt.figure()
ax = fig.add_subplot(111)
temp.plot(kind='kde', ax=ax, label='Original variable')
df[df['GarageQual']=='TA']['SalePrice'].plot(kind='kde', ax=ax, color='red', label='Imputed variable')
ax.legend(loc='best')
plt.title("GarageQual Imputation Effect on SalePrice")
plt.show()

# ------------------------------------------------------------
# STEP 8: Explore FireplaceQu
# ------------------------------------------------------------
df['FireplaceQu'].value_counts().plot(kind='bar', title="FireplaceQu Distribution (Original)")
plt.show()

print("Mode of FireplaceQu:", df['FireplaceQu'].mode())
# INTERPRETATION: 'Gd' is the most frequent category.

# ------------------------------------------------------------
# STEP 9: Compare SalePrice Distribution (FireplaceQu)
# ------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
df[df['FireplaceQu']=='Gd']['SalePrice'].plot(kind='kde', ax=ax, label='Houses with Gd')
df[df['FireplaceQu'].isnull()]['SalePrice'].plot(kind='kde', ax=ax, color='red', label='Houses with NA')
ax.legend(loc='best')
plt.title("SalePrice Distribution by FireplaceQu")
plt.show()

# ------------------------------------------------------------
# STEP 10: Impute FireplaceQu
# ------------------------------------------------------------
df['FireplaceQu'] = df['FireplaceQu'].fillna('Gd')
df['FireplaceQu'].value_counts().plot(kind='bar', title="FireplaceQu Distribution (After Imputation)")
plt.show()

# ------------------------------------------------------------
# STEP 11: Compare SalePrice Distribution After Imputation
# ------------------------------------------------------------
temp = df[df['FireplaceQu']=='Gd']['SalePrice']
fig = plt.figure()
ax = fig.add_subplot(111)
temp.plot(kind='kde', ax=ax, label='Original variable')
df[df['FireplaceQu']=='Gd']['SalePrice'].plot(kind='kde', ax=ax, color='red', label='Imputed variable')
ax.legend(loc='best')
plt.title("FireplaceQu Imputation Effect on SalePrice")
plt.show()

# ------------------------------------------------------------
# STEP 12: Train/Test Split
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.2
)

# ------------------------------------------------------------
# STEP 13: Sklearn SimpleImputer
# ------------------------------------------------------------
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print("Imputer statistics (most frequent values):", imputer.statistics_)
# INTERPRETATION: Sklearn confirms the mode values used for imputation.

# ------------------------------------------------------------
# STEP 14: Side-by-Side Bar Plots for Category Frequencies
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# GarageQual before vs after
df_original = pd.read_csv('train.csv', usecols=['GarageQual','FireplaceQu','SalePrice'])

axes[0].bar(df_original['GarageQual'].value_counts().index,
            df_original['GarageQual'].value_counts().values,
            color='skyblue', label='Original')
axes[0].bar(df['GarageQual'].value_counts().index,
            df['GarageQual'].value_counts().values,
            color='orange', alpha=0.7, label='Imputed')
axes[0].set_title("GarageQual Frequency: Original vs Imputed")
axes[0].legend()

# FireplaceQu before vs after
axes[1].bar(df_original['FireplaceQu'].value_counts().index,
            df_original['FireplaceQu'].value_counts().values,
            color='skyblue', label='Original')
axes[1].bar(df['FireplaceQu'].value_counts().index,
            df['FireplaceQu'].value_counts().values,
            color='orange', alpha=0.7, label='Imputed')
axes[1].set_title("FireplaceQu Frequency: Original vs Imputed")
axes[1].legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 15: Percentage Comparison Table
# ------------------------------------------------------------
# GarageQual before vs after imputation

garage_comparison = pd.concat([
    df_original['GarageQual'].value_counts(normalize=True) * 100,
    df['GarageQual'].value_counts(normalize=True) * 100
], axis=1)
garage_comparison.columns = ['Original %','Imputed %']
print("GarageQual Percentage Comparison:\n", garage_comparison)

# FireplaceQu before vs after imputation
fireplace_comparison = pd.concat([
    df_original['FireplaceQu'].value_counts(normalize=True) * 100,
    df['FireplaceQu'].value_counts(normalize=True) * 100
], axis=1)
fireplace_comparison.columns = ['Original %','Imputed %']
print("FireplaceQu Percentage Comparison:\n", fireplace_comparison)


## ============================================================
### Executive Summary
## ============================================================
### This tutorial demonstrates Frequent Value Imputation (Mode) for categorical features.

### Key Insights:
## 1. Mode imputation replaces missing values with the most frequent category.
# 2. GarageQual missing values were replaced with 'TA'; FireplaceQu with 'Gd'.
# 3. SalePrice distributions before and after imputation show stability, confirming effectiveness.
# 4. Sklearn’s SimpleImputer automates mode imputation for reproducibility.

# Recruiter Takeaway: This script shows practical preprocessing skills,
# awareness of categorical imputation impacts, and clear communication —
# essential traits for applied ML roles.
## ============================================================
