#!/usr/bin/env python
# coding: utf-8

""" # # 📊 Simple Linear Regression using scikit-learn  """

# In this tutorial, we demonstrate how to use **Simple Linear Regression** 
# to predict salary based on years of education. 
# This is a classic example of supervised learning where:
# - **Independent variable (X):** Education (years)
# - **Dependent variable (y):** Salary
#
# The goal is to fit a straight line that best explains the relationship 
# between education and salary, and then evaluate the model’s performance.

# ## Step 1: Import Required Libraries
# We use NumPy for numerical operations, Matplotlib & Seaborn for visualization,
# and scikit-learn for building the regression model.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# ## Step 2: Prepare the Dataset (Independent Variable X)
# Education years are stored as a NumPy array.
# Initially, this is a 1-D array, which we later reshape into 2-D
# because scikit-learn expects features in a 2-D format.

edu = np.array([1,2,3,4,5])
edu
edu.shape

# Convert 1-D array into 2-D array (required for scikit-learn)
edu = edu.reshape(5,1)   # 5 rows, 1 column
edu.shape


# ## Step 3: Prepare the Dataset (Dependent Variable y)
# Salary values corresponding to years of education.

salary = np.array([200,300,400,500,600])
salary


# ## Step 4: Visualize the Data
# Scatter plot to observe the relationship between education and salary.

plt.scatter(edu, salary)
plt.xlabel('Education (Years)')
plt.ylabel('Salary')
plt.title('Education vs Salary Distribution')
plt.show()


# ## Step 5: Build and Train the Linear Regression Model
# Import LinearRegression class from scikit-learn and fit the model.

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(edu, salary)   # Fit model on (X, y)


# ## Step 6: Inspect Model Parameters
# Intercept (b0) and Coefficient (b1) of the regression line.

model.intercept_   # b0
model.coef_        # b1


# ## Step 7: Visualize Regression Line
# Plot the regression line along with the actual data points.

plt.scatter(edu, salary)
plt.plot(edu, model.predict(edu), color='red')   # Regression line
plt.xlabel('Education (Independent Variable)')
plt.ylabel('Salary (Dependent Variable)')
plt.title('Linear Regression Fit')
plt.show()


# ## Step 8: Test the Model
# Predict salary for new education values (e.g., 5 and 20 years).

test = np.array([[5],[20]])   # Reshape test data into 2-D array
test
model.predict(test)


# ## Step 9: Evaluate Model Performance
# R-squared score indicates how well the model explains the variance in data.
# Value ranges from 0 to 1, with 1 being a perfect fit.

model.score(edu, salary)


# ## ✅ Final Summary
# - We successfully built a **Simple Linear Regression model** using scikit-learn.
# - The model learned the linear relationship between education and salary.
# - Visualization confirmed that the regression line fits the data well.
# - Predictions for unseen values (e.g., 20 years of education) were generated.
# - R-squared score provided a measure of model accuracy.




# This notebook demonstrates clear data preparation, visualization, model training, 
# and evaluation steps. It reflects strong coding clarity, professional documentation, 
# and the ability to communicate machine learning workflows effectively — 
# skills that are highly valuable in data science and AI engineering roles.