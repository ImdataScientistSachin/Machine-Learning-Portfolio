#!/usr/bin/env python
# coding: utf-8

"""   # # K-Nearest Neighbour Regression     """


# #### K-Nearest Neighbors (KNN) regression is a non-parametric method used for predicting continuous outcomes based on the values of nearby data points in the feature space. 

# #### It is an extension of the KNN algorithm used for classification tasks, but instead of assigning a class label, it predicts a numerical value.


""" Practical Implementation of K-Nearest Neighbour Regression  """

# loading the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Loading the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset


# visualizing the dataset

plt.scatter(dataset['Level'],dataset['Salary'])
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


# LoDing the feature and target variables
X = dataset.iloc[:,[1]].values
X


# DEfining the target variable
Y = dataset.iloc[:,2].values
Y



# import the KNN packages

from sklearn.neighbors import KNeighborsRegressor

# create the model
model = KNeighborsRegressor(n_neighbors=3)

# fit the model
model.fit(X,Y)

# evaluate the model
model.score(X,Y)

# predicting the values
Y_pred = model.predict(X)
Y_pred

# manually prediction
model.predict(np.array([[6.4]]))

# Visualizing the KNN regression results
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,model.predict(X_grid),color='blue')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# In conclusion, K-Nearest Neighbors (KNN) regression is a straightforward and effective technique for predicting continuous outcomes based on the proximity of data points in the feature space.
# The model captures the non-linear relationship between Level and Salary effectively by averaging the salaries of the nearest neighbors.
# In KNN regression, the choice of 'k' (the number of neighbors) is crucial, as it influences the model's bias-variance trade-off. A small 'k' may lead to overfitting, while a large 'k' can smooth out important patterns in the data.
# Cross-validation techniques can be employed to select the optimal 'k' value for better performance.
# KNN regression is sensitive to the scale of the features, so feature scaling (e.g., normalization or standardization) is often necessary to ensure that all features contribute equally to the distance calculations.
# Additionally, KNN regression can be computationally expensive for large datasets, as it requires calculating distances to all training points for each prediction.
# Overall, KNN regression is a simple yet effective method for regression tasks, especially when the relationship between features and the target variable is complex and non-linear.