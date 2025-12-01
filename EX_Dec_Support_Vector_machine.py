#!/usr/bin/env python
# coding: utf-8

""" 🍦🍨 # Support Vector Regression (SVR) on Ice Cream Sales Dataset ☀️ """

# This script applies Support Vector Regression (SVR) to model the relationship
# between daily temperature and ice cream sales, experimenting with different
# kernel functions to find the best fit.

## 📚 Import Essential Libraries
# Load the necessary tools for numerical operations, data handling, and visualization.

import numpy as np # For high-performance numerical computations
import pandas as pd # For reading and manipulating the dataset (DataFrames)
import matplotlib.pyplot as plt # For creating static, interactive, and animated visualizations
import seaborn as sns # For making statistical graphics more attractive and informative
sns.set_style('whitegrid') # Setting a clean background style for our plots

## 🧊 Load the Dataset
# We are reading the 'Ice_cream_selling_data.csv' file into a pandas DataFrame.
# This dataset is simple, featuring 'Temperature' as the independent variable (X)
# and 'Ice Cream Sales' as the dependent variable (Y).

dataset = pd.read_csv('Ice_cream_selling_data.csv')
# Displaying the first few rows of the dataset to verify successful loading and structure.
dataset

## 📈 Visualize the Data Relationship
# Let's quickly plot the raw data to see the underlying pattern.
# We expect to see a positive, likely non-linear, correlation between temperature and sales.

plt.scatter(dataset['Temperature (°C)'],dataset['Ice Cream Sales (units)'])
plt.xlabel('Temperature (°C)') # X-axis: The factor influencing sales
plt.ylabel('Ice Cream Sales') # Y-axis: The outcome we want to predict
plt.title('Temperature vs. Ice Cream Sales') # A descriptive title for the plot
plt.show()

## 🔄 Prepare Data for SVR Model
# SVR models typically expect the feature matrix (X) to be a 2D array,
# even if it contains only one feature. We reshape it for compatibility.

# Independent variable (Feature): Temperature
X = dataset.iloc[:,[0]].values # Selects all rows (:) and the first column ([0]) for Temperature
# X is now a 2D NumPy array, ready for training.
X

# Dependent variable (Target): Ice Cream Sales
Y = dataset.iloc[:,1].values # Selects all rows (:) and the second column ([1]) for Sales
# Y is a 1D NumPy array of the target values.
Y

## ⚙️ Import the SVR Model
# Importing the Support Vector Regression class from scikit-learn's SVM module.

from sklearn.svm import SVR

## 🧠 Instantiate Multiple SVR Models with Different Kernels
# SVR's power lies in its kernel trick, allowing it to find non-linear relationships.
# We test four common kernels: RBF (Radial Basis Function), Sigmoid, Polynomial, and Linear.

# Model 1: RBF Kernel (a good default for non-linear data)
model = SVR(kernel='rbf', degree=10) # 'degree' parameter is mostly ignored for 'rbf'

# Model 2: Sigmoid Kernel (can perform well in certain non-linear separation scenarios)
model_1 = SVR(kernel='sigmoid', degree=8) # 'degree' is ignored here too

# Model 3: Polynomial Kernel (suitable for data with a curved relationship)
model_2 = SVR(kernel='poly', degree=8) # 'degree' specifies the polynomial's power

# Model 4: Linear Kernel (simple linear boundary, often a baseline comparison)
model_3 = SVR(kernel='linear', degree=8) # 'degree' is ignored for 'linear'

## 🚀 Train the Models
# The 'fit' method trains the SVR model using the Temperature (X) to predict Sales (Y).

print("Training SVR models...")

model.fit(X,Y) # Training RBF Model

model_1.fit(X,Y) # Training Sigmoid Model

model_2.fit(X,Y) # Training Polynomial Model

model_3.fit(X,Y) # Training Linear Model

print("Models trained successfully!")

## 🎯 Evaluate Model Performance
# We use the 'score' method, which returns the coefficient of determination (R^2)
# for the prediction of Y from X. A value closer to 1.0 indicates a better fit.

print("\n--- Model R-squared (R²) Scores ---")

# RBF Model Score
rbf_score = model.score(X,Y)
print(f"RBF Kernel Model Score: {rbf_score:.4f}")

# Sigmoid Model Score
sigmoid_score = model_1.score(X,Y)
print(f"Sigmoid Kernel Model Score: {sigmoid_score:.4f}")

# Polynomial Model Score
poly_score = model_2.score(X,Y)
print(f"Polynomial Kernel Model Score: {poly_score:.4f}")

# Linear Model Score
linear_score = model_3.score(X,Y)
print(f"Linear Kernel Model Score: {linear_score:.4f}")

# The kernel with the highest R² score is generally the best fit for the data.
print("\n--- Evaluation Summary ---")
print(f"The best performing model on this training data is likely the one with the highest score (RBF: {rbf_score:.4f}, Sigmoid: {sigmoid_score:.4f}, Polynomial: {poly_score:.4f}, Linear: {linear_score:.4f}).")