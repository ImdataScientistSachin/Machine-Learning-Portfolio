#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Handling Missing Data - Complete Case Analysis (CCA)
#
# Definition:
# Complete Case Analysis removes rows with missing values.
# It is appropriate when missingness is < 5% and assumed random.
#
# Objective:
# Demonstrate how CCA affects numerical distributions and categorical proportions.
#
# Why Important:
# - Simple and transparent method.
# - Avoids introducing artificial values.
# - But reduces dataset size and may bias results if missingness is not random.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Load Libraries
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 2: Load Dataset
# ------------------------------------------------------------
dataset = pd.read_csv('data_science_job.csv')
print("Initial dataset sample:\n", dataset.head())

# ------------------------------------------------------------
# STEP 3: Explore Missingness
# ------------------------------------------------------------
print("Percentage of missing values per column:\n", dataset.isnull().mean()*100)
# INTERPRETATION:

# gender = 23% missing
# major_discipline = 14% missing
# company_size = 31% missing
# company_type = 32% missing
# These are too high for CCA. We will only apply CCA to columns with <5% missingness.

print("Dataset shape:", dataset.shape)

# ------------------------------------------------------------
# STEP 4: Select Columns with <5% Missingness
# ------------------------------------------------------------

cols = [var for var in dataset.columns if dataset[var].isnull().mean() < 0.05 and dataset[var].isnull().mean() > 0]
print("Columns eligible for CCA (<5% missing):", cols)

print("Sample of selected columns:\n", dataset[cols].sample(5))

print("Education level categories:\n", dataset['education_level'].value_counts())

# ------------------------------------------------------------
# STEP 5: Apply Complete Case Analysis
# ------------------------------------------------------------
new_dts = dataset[cols].dropna()
print("Original vs CCA dataset shapes:", dataset.shape, new_dts.shape)

print("Proportion of complete cases retained:", len(new_dts)/len(dataset))
# INTERPRETATION: Only a small fraction of rows are dropped, so CCA is safe here.

# len(dataset[cols].dropna()): Counts the number of rows remaining after removing those with missing values.
# / len(dataset): Divides the number of complete rows by the total number of rows to get the proportion of complete cases.

# ------------------------------------------------------------
# STEP 6: Compare Numerical Distributions
# ------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
dataset['training_hours'].hist(bins=50, ax=ax, density=True, color='red', alpha=0.6, label='Original')
new_dts['training_hours'].hist(bins=50, ax=ax, density=True, color='green', alpha=0.6, label='CCA')
ax.legend()
plt.title("Distribution of Training Hours: Original vs CCA")
plt.show()
# INTERPRETATION: Training hours distribution remains similar, showing CCA did not distort this variable.

fig = plt.figure()
ax = fig.add_subplot(111)
dataset['city_development_index'].plot.density(color='red', label='Original')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_dts['city_development_index'].plot.density(color='green', label='CCA')
ax.legend()
plt.title("Density of City Development Index: Original vs CCA")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
dataset['experience'].plot.density(color='red', label='Original')
new_dts['experience'].plot.density(color='green', label='CCA')
ax.legend()
plt.title("Density of Experience: Original vs CCA")
plt.show()
# INTERPRETATION: Distributions overlap closely, confirming CCA did not bias numerical features.

# ------------------------------------------------------------
# STEP 7: Compare Categorical Distributions
# ------------------------------------------------------------

temp = pd.concat([
    dataset['enrolled_university'].value_counts() / len(dataset),
    new_dts['enrolled_university'].value_counts() / len(new_dts)
], axis=1)
temp.columns = ['original','cca']
print("Enrolled University distribution comparison:\n", temp)
# INTERPRETATION: Category proportions remain stable, showing CCA did not distort categorical balance.

temp = pd.concat([
    dataset['education_level'].value_counts() / len(dataset),
    new_dts['education_level'].value_counts() / len(new_dts)
], axis=1)
temp.columns = ['original','cca']
print("Education Level distribution comparison:\n", temp)

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------

print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This tutorial demonstrates Complete Case Analysis (CCA) for handling missing data.")
print("Key Insights:")
print("1. CCA is appropriate when missingness <5% and random.")
print("2. Numerical distributions (training_hours, city_development_index, experience) remain stable after CCA.")
print("3. Categorical proportions (enrolled_university, education_level) are preserved.")
print("4. Dataset size is reduced slightly, but without major distortion.")
print("\nRecruiter Takeaway: This script shows practical preprocessing skills,")
print("awareness of statistical impacts, and clear communication — essential traits for applied ML roles.")
print("============================================================")