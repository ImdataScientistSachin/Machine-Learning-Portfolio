#!/usr/bin/env python
# coding: utf-8

"""  #  Exercise Day 2 – Scikit-Learn Example 3  
# ## Predicting CGPA using Attendance and SAT Scores  """ 
#
# In this exercise, we build a simple regression model to predict a student's GPA
# based on two independent variables: SAT scores and class attendance.
# The workflow demonstrates data preprocessing, visualization, and model training
# using scikit-learn. This notebook is structured to be both educational and

# ---------------------------------------------------------
# Step 1: Import and configure libraries
# ---------------------------------------------------------
# NumPy and Pandas for numerical and tabular data handling
# Matplotlib and Seaborn for visualization
# Statsmodels for statistical modeling (optional, not used directly here)
# Seaborn style is set to 'whitegrid' for cleaner plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set_style('whitegrid')


# ---------------------------------------------------------
# Step 2: Load dataset
# ---------------------------------------------------------
# The dataset is assumed to be stored in 'dummy.csv'.
# It contains columns for SAT scores, GPA, and Attendance.

dataset = pd.read_csv('dummy.csv')
dataset


# ---------------------------------------------------------
# Step 3: Preprocess categorical variables
# ---------------------------------------------------------
# Attendance is stored as 'Yes' or 'No'. We convert this into
# numerical form (Yes=1, No=0) so it can be used in regression.

dataset['Attendance'] = dataset['Attendance'].map({'No': 0, 'Yes': 1})
dataset


# ---------------------------------------------------------
# Step 4: Visualize relationships
# ---------------------------------------------------------
# Scatter plot of SAT scores vs GPA
# This helps us visually inspect whether SAT scores correlate with GPA.

plt.scatter(dataset['SAT'], dataset['GPA'])
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.title('Relationship between SAT Scores and GPA')
plt.show()


# Scatter plot of Attendance vs GPA
# Independent variables are always plotted on the X-axis.
# This shows how attendance impacts GPA.

plt.scatter(dataset['Attendance'], dataset['GPA'])
plt.xlabel('Attendance (0 = No, 1 = Yes)')
plt.ylabel('GPA')
plt.title('Relationship between Attendance and GPA')
plt.show()


# ---------------------------------------------------------
# Step 5: Define independent and dependent variables
# ---------------------------------------------------------
# X contains SAT scores and Attendance (independent variables).
# Y contains GPA (dependent variable).

X = dataset.iloc[:, [0, 2]].values
X

X.shape

Y = dataset.iloc[:, 1].values
Y


# ---------------------------------------------------------
# Step 6: Train Linear Regression model
# ---------------------------------------------------------
# We import LinearRegression from scikit-learn and fit the model
# using SAT and Attendance as predictors for GPA.

from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Fit the model on the dataset
model.fit(X, Y)


# ---------------------------------------------------------
# Step 7: Inspect model parameters
# ---------------------------------------------------------
# The intercept (b0) represents the baseline GPA when predictors are zero.
# The coefficients (b1, b2) represent the effect of SAT and Attendance respectively.

model.intercept_

model.coef_


# ---------------------------------------------------------
# Step 8: Evaluate model performance
# ---------------------------------------------------------
# The score method returns the R-squared value, which indicates
# how well the independent variables explain the variance in GPA.

model.score(X, Y)


# ---------------------------------------------------------
# Final Summary – Key Takeaways
# ---------------------------------------------------------
# This exercise demonstrated how to use SAT scores and Attendance
# as predictors for GPA using a Linear Regression model.
#
# ✅ Data preprocessing:
#    - Converted categorical Attendance values ('Yes'/'No') into numeric form.
#
# ✅ Visualization:
#    - Scatter plots revealed positive correlation between SAT and GPA.
#    - Attendance also showed a clear impact on GPA distribution.
#
# ✅ Model training:
#    - Linear Regression was applied with SAT and Attendance as independent variables.
#    - Intercept and coefficients quantified the baseline GPA and predictor effects.
#
# ✅ Model evaluation:
#    - R-squared score indicated how well SAT and Attendance explain GPA variance.
#
# Overall, this workflow highlights the importance of combining academic performance
# (SAT scores) with behavioral factors (Attendance) to predict student outcomes.
# The notebook serves as a clear, recruiter-ready demonstration of data science
# skills: preprocessing, visualization, modeling, and interpretation.