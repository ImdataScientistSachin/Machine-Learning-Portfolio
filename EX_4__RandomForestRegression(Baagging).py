#!/usr/bin/env python
# coding: utf-8

"""    # Random Forest Regression(Ensemble_Technique - Baagging)   """
# ## Random Forest Regression
# ### Random Forest is an ensemble learning method that combines multiple decision trees to improve predictive performance and reduce overfitting.

# ### It operates by constructing a multitude of decision trees during training and outputs the average prediction of the individual trees for regression tasks.

# ### Each tree is built using a random subset of the data and features, which helps to create diversity among the trees and enhances the overall model's robustness.

# #### Random Forest Regression is an extension of the Random Forest algorithm, which combines multiple decision trees to enhance prediction accuracy and control overfitting.

# #### It operates by constructing a multitude of decision trees during training and outputs the average prediction of the individual trees for regression tasks.


"""  Practical  Implemetation  of  Random Forest Regression  """
# Practical Example: position_Salaries dataset to predict the salary based on the level of position using Random Forest Regression.


# Load  the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


#  load the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset

# Convertion dataset into 2 D
X = dataset.iloc[:,[1]].values
X

# Target Variable
Y = dataset.iloc[:,[2]].values
Y


# Plot the distribution 

plt.scatter(dataset['Level'],dataset['Salary'])
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


# load the Random forest library
from sklearn.ensemble import RandomForestRegressor


# prepare the model 
model = RandomForestRegressor(n_estimators=11)

# n_estimators: The number of trees in the forest.
# max_depth: The maximum depth of each tree (optional).
# min_samples_split: The minimum number of samples required to split an internal node (optional).
# random_state: Controls the randomness of the estimator (optional).

# Fit the model
model.fit(X,Y)

# Finding the model prediction
model.predict(np.array([[6.5]]))

# Finding the model prediction score
model.score(X,Y)

# prepare the distribution for higher accuracy
X_grid = np.arange(1,10,0.01)
X_grid

# reshape the data
X_grid = X_grid.reshape(-1,1)


# The first parameter -1 indicates that NumPy should automatically determine the size of this dimension based 
#  on the total number of elements in X_grid.
# The second parameter 1 specifies that the reshaped array should have one column.
X_grid


# prepare the distribution 

plt.scatter(dataset['Level'],dataset['Salary'])
plt.plot(X_grid,model.predict(X_grid),color='Green')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# The scatter plot shows the original data points (Level vs. Salary) from the dataset.
# The green line represents the predictions made by the Random Forest Regression model across the range of levels
# (from 1 to 10) using the fine-grained X_grid.
# This visualization helps to illustrate how well the Random Forest model fits the data and captures the underlying
# relationship between the level of position and the corresponding salary.
# The use of X_grid with small increments (0.01) allows for a smoother curve, providing a clearer picture of the model's behavior.