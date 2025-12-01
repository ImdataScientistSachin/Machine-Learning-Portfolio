
#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Executive Summary: Data Preprocessing in Machine Learning
# ============================================================

#  Data Preprocessing are used to  transformation of raw data into a suitable format for model training. This process is essential for enhancing the accuracy and performance of machine learning algorithms.

#  I) One-hot encoding :  is a method of representing categorical variables as binary vectors. Each category is transformed into a vector where only one element is "hot" (set to 1) and all other elements are "cold" (set to 0).

# ##### Red: [1, 0, 0]
# ##### Green: [0, 1, 0]
# ##### Blue: [0, 0, 1]

# ##### II ) # Features scaling : It's a technnique to standardize the independent features present in the data in a fixed range. Convert large range Independent dataset to small range.

# ##### A ) standardization : particularly useful for preparing features for algorithms that are sensitive to the scale of input data.Also knows as Z-score Normalization.

# ##### B ) Normalization : 



# Purpose:
# This script demonstrates essential preprocessing techniques in machine learning:
# - Encoding categorical variables (Label Encoding, One-Hot Encoding)
# - Feature scaling (Standardization, MinMaxScaler, MaxAbsScaler)
# - Handling missing values (Imputation)

# Why it matters:
# Preprocessing transforms raw data into a suitable format for model training.
# It enhances accuracy, ensures fair feature contribution, and improves performance
# of algorithms sensitive to scale or categorical representation.

# Dataset:
# Iris_Updated.csv — updated Iris dataset with renamed columns.

# Workflow:
# 1. Load dataset
# 2. Encode categorical variables
# 3. Apply feature scaling
# 4. Handle missing values
# 5. Summarize results

# ============================================================

# ==== Step 1: Import Libraries ====

# NumPy and Pandas for data handling, Matplotlib/Seaborn for visualization.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # clean, professional plot style

# ==== Step 2: Load Dataset ====

# Load Iris dataset and rename columns for clarity.
dataset = pd.read_csv('Iris_Updated.csv',
                      names=['Sep_Len','Sep_Wid','Peb_Len','Peb_Wid','Label'])
print(dataset.head())

# ==== Step 3: Separate Features and Labels ====

# X = feature matrix (all columns except last)
# Y = target labels (last column)
X = dataset.iloc[:,0:-1].values
Y = dataset.iloc[:,-1].values
print("Feature matrix shape:", X.shape)
print("Target vector shape:", Y.shape)

# ==== Step 4: Label Encoding ====

# LabelEncoder converts categorical labels into numeric codes.
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
T_LE = LE.fit_transform(Y)
print("Label Encoded values:", T_LE)

# ==== Step 5: One-Hot Encoding ====

# OneHotEncoder converts categorical labels into binary vectors.
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(sparse_output=False)  # dense array output
T_OHE = OHE.fit_transform(Y.reshape(-1,1))  # reshape to 2D
print("One-Hot Encoded values:\n", T_OHE)


# ==== Step 6: Feature Scaling - Standardization ====

# StandardScaler standardizes features to mean=0, variance=1 (Z-score normalization).
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print("Standardized features:\n", X_std)

# Visualize distribution before and after scaling
sns.distplot(X[:,0]); plt.title("Original Feature Distribution"); plt.show()
sns.distplot(X_std[:,0]); plt.title("Standardized Feature Distribution"); plt.show()

# ==== Step 7: Feature Scaling - MinMaxScaler ====

# MinMaxScaler normalizes features to a specified range (default [0,1]).
from sklearn.preprocessing import MinMaxScaler
mmScaler = MinMaxScaler(feature_range=(1,100))  # custom range
X_mm = mmScaler.fit_transform(X)
print("MinMax Scaled features:\n", X_mm)

# ==== Step 8: Feature Scaling - MaxAbsScaler ====

# MaxAbsScaler scales features by their maximum absolute value.
from sklearn.preprocessing import MaxAbsScaler
MAScaler = MaxAbsScaler()
X_mab = MAScaler.fit_transform(X)
print("MaxAbs Scaled features:\n", X_mab)

# ==== Step 9: Handling Missing Values ====

# SimpleImputer replaces missing values using strategies: mean, median, or most_frequent (mode).
test = np.array([[1,3],[7,5],[7,3],[np.nan,2],[np.nan,np.nan]])
print("Dummy data with missing values:\n", test)

from sklearn.impute import SimpleImputer
# Choose strategy: 'mean', 'median', or 'most_frequent'

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
print("Imputed data:\n", imp.fit_transform(test))


# ============================================================
# Final Summary (Recruiter-Friendly Wrap-Up)
# ============================================================

# Preprocessing Techniques Demonstrated:
# - Label Encoding: converted categorical labels into integers.
# - One-Hot Encoding: represented categories as binary vectors.
# - Standardization: scaled features to mean=0, variance=1.
# - MinMaxScaler: normalized features to a custom range [1,100].
# - MaxAbsScaler: scaled features by maximum absolute value.
# - Imputation: handled missing values using mode replacement.
#
# Key Takeaway:
# Data preprocessing ensures that machine learning models receive clean,
# consistent, and properly scaled input. This improves accuracy, stability,
# and interpretability of results. The script is self-documenting and
# recruiter-friendly, showing both technical workflow and explanatory narrative.
# ============================================================