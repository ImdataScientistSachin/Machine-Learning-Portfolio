#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Predicting the Price of Used Cars** using multiple regression techniques and data preprocessing steps.
# ================================================================

# This project demonstrates **Predicting the Price of Used Cars** using
# multiple regression techniques and data preprocessing steps.

# Purpose:
# - To build a regression model that predicts car prices based on features
#   such as brand, year, engine size, mileage, and body type.
# - To showcase the full workflow: data cleaning, handling missing values,
#   outlier removal, feature engineering, multicollinearity checks, dummy
#   variable creation, scaling, model training, and evaluation.

# Why it matters:
# - Predicting car prices is a practical business problem with applications
#   in dealerships, resale platforms, and financial services.
# - Recruiters value candidates who can demonstrate end-to-end data science
#   pipelines, statistical rigor, and clear communication of results.

# Techniques highlighted:
# - Handling missing values and outliers.
# - Log transformation to normalize skewed distributions.
# - Variance Inflation Factor (VIF) to detect multicollinearity.
# - Dummy variable encoding for categorical features.
# - Feature scaling with StandardScaler.
# - Linear regression modeling and evaluation.

# ================================================================


# ==== Step 1: Import Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm


# ==== Step 2: Load Dataset ====
dataset = pd.read_csv('reallifedata.csv')
dataset.describe()              # Summary of numerical data
dataset.describe(include='all') # Summary including categorical data
dataset.Brand.unique()          # Unique car brands
dataset.Body.unique()           # Unique body types
dataset.Model.unique()          # Unique models

# Drop 'Model' column (not useful for prediction)
data = dataset.drop('Model', axis=1)


# ==== Step 3: Handle Missing Values ====
# Check for missing values
data.isnull().sum()

# Drop rows with missing values (alternative: impute with mean/median/mode)
data_no_mv = data.dropna(axis=0)
data_no_mv.isnull().sum()  # Confirm no missing values remain


# ==== Step 4: Outlier Detection and Removal ====
# Distribution of Price shows exponential skew with outliers
sns.distplot(data_no_mv['Price'])
plt.show()

# Remove top 4% extreme values in Price
Q = data_no_mv['Price'].quantile(0.96)
data_1 = data_no_mv[data_no_mv['Price'] < Q]

# Remove top 5% extreme values in Mileage
Q = data_1['Mileage'].quantile(0.95)
data_2 = data_1[data_1['Mileage'] < Q]

# Remove unrealistic EngineV values (>6 liters)
data_3 = data_2[data_2['EngineV'] < 6]

# Remove cars older than 5th percentile year
Q = data_3['Year'].quantile(0.05)
data_4 = data_3[data_3['Year'] > Q]

# Reset index after cleaning
data_cleaned = data_4.reset_index(drop=True)


# ==== Step 5: Log Transformation ====
# Price distribution is skewed → apply log transformation
log_price = np.log(data_cleaned['Price'])
sns.distplot(log_price)
plt.show()

# Replace Price with log_price for linearity
data_cleaned = data_cleaned.drop('Price', axis=1)
data_cleaned['log_price'] = log_price


# ==== Step 6: Multicollinearity Check ====
# Correlation matrix
test = data_cleaned[['Mileage', 'EngineV', 'Year']]
test.corr()

# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(test.values, i) for i in range(3)]
# Interpretation: High VIF indicates multicollinearity.
# Year shows high VIF → drop it.
data_no_multicolinearity = data_cleaned.drop('Year', axis=1)


# ==== Step 7: Handle Categorical Variables ====
# Convert categorical variables (Brand, Body, Registration) into dummy variables
data_with_dummies = pd.get_dummies(data_no_multicolinearity, drop_first=True, dtype=int)

# Define features (X) and target (y)
X = data_with_dummies.drop('log_price', axis=1)
y = data_with_dummies['log_price']


# ==== Step 8: Feature Scaling ====
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Distribution after scaling
sns.distplot(X_scaled[:, 0])
plt.show()


# ==== Step 9: Train-Test Split ====
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)


# ==== Step 10: Train Linear Regression Model ====
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Model R² Score:", model.score(X_test, y_test))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual log_price')
plt.ylabel('Predicted log_price')
plt.title('Actual vs Predicted Prices')
plt.show()

# Convert predictions back to original price scale
y_pred = np.exp(y_pred)
y_pred


# ================================================================
# FINAL SUMMARY
# ================================================================

# - We built a regression model to predict used car prices.
# - Data preprocessing included handling missing values, removing outliers,
#   log transformation, multicollinearity checks, and dummy variable encoding.
# - StandardScaler ensured features were normalized.
# - Linear regression achieved a reasonable R² score, showing predictive power.
# - Predictions were converted back to original price scale for interpretation.
#

# ================================================================