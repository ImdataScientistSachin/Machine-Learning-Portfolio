#!/usr/bin/env python
# coding: utf-8

# #   Multiple LINEAR Regression Practice Session 3  
# ---------------------------------------------------------

# In this notebook, we will perform **Multiple Linear Regression** using dummy data.  
# The goal is to understand how SAT scores and Attendance influence GPA.  
# We'll use Python libraries like NumPy, Pandas, Matplotlib, Seaborn, and Statsmodels.

# ##  Dummy data  


# Import the essential libraries for numerical computation, data handling, and visualization.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')   # Set a clean background style for plots



# Load the dataset (dummy.csv) into a Pandas DataFrame.
dataset = pd.read_csv('dummy.csv')



# Display the dataset to get an initial look at the data.
dataset


# Convert the 'Attendance' column (categorical: Yes/No) into numerical format (1/0).
# This transformation is necessary for regression models which require numerical inputs.
dataset['Attendance'] = dataset['Attendance'].map({'No':0,'Yes':1})



# Verify that the 'Attendance' column has been successfully converted.
dataset



# Plot the relationship between SAT scores and GPA.
# This helps us visually inspect whether SAT scores correlate with GPA.
plt.scatter(dataset['SAT'], dataset['GPA'])
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


# Plot the relationship between Attendance and GPA.
# Independent variables (SAT, Attendance) are always placed on the X-axis.
plt.scatter(dataset['Attendance'], dataset['GPA'])
plt.xlabel('ATTENDANCE')
plt.ylabel('GPA')
plt.show()



# Define the independent variables (X): SAT and Attendance.
X = dataset[['SAT','Attendance']]



# Display the independent variables to confirm selection.
X



# Define the dependent variable (Y): GPA.
Y = dataset['GPA']



# Display the dependent variable.
Y



# Import Statsmodels library to build and train the regression model.
import statsmodels.api as sm



# Add a constant term to the independent variables.
# This represents the intercept in the regression equation.
X1 = sm.add_constant(X)

# Display the dataset with the added constant column.
X1

# Train the regression model using Ordinary Least Squares (OLS).
model = sm.OLS(Y, X1).fit()

# Display the summary of the regression model.
# This includes coefficients, R-squared value, p-values, and other statistical metrics.
model.summary()

# Test the model with new data.
# Example: SAT = 1700, Attendance = Yes (1) and No (0).
test = pd.DataFrame([[1.0,1700,1],[1.0,1700,0]], columns=['Const','SAT','Attendance'])

# Display the test dataset.
test

# Predict GPA using the trained model for the test dataset.
model.predict(test)



# Manual prediction using regression equation:
# GPA = Intercept + (SAT coefficient * SAT) + (Attendance coefficient * Attendance)
Y_hat_yes = 0.6439 + 0.0014*dataset['SAT'] + 0.2226*1   # For attending students
Y_hat_no  = 0.6439 + 0.0014*dataset['SAT'] + 0.2226*0   # For non-attending students


# Display predicted GPA values for attending students.
Y_hat_yes

# Display predicted GPA values for non-attending students.
Y_hat_no 


# Plot the best-fitting regression line for attending students.
plt.scatter(dataset['SAT'], dataset['GPA'])
plt.plot(dataset['SAT'], Y_hat_yes, color='Purple')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()



# Plot the best-fitting regression line for non-attending students.
plt.scatter(dataset['SAT'], dataset['GPA'])
plt.plot(dataset['SAT'], Y_hat_no, color='Red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()

# Compare both regression lines (attending vs non-attending students).
plt.scatter(dataset['SAT'], dataset['GPA'])
plt.plot(dataset['SAT'], Y_hat_yes, color='green', label='Attending')
plt.plot(dataset['SAT'], Y_hat_no, color='red', label='Non-Attending')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.legend()
plt.show()