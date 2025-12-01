#!/usr/bin/env python
# coding: utf-8
 
"""  ## Decision Tree Regression    """ 
# ### Decision Tree (DT) is a supervised machine learning algorithm used for both regression and classification tasks.
 
# #### DT operate by splitting data into subsets based on feature values, forming a tree-like structure that aids in decision-making.

# ###  A decision tree consists of: 
# ####   Root Node:   The starting point that represents the entire dataset.
# ####   Internal Nodes: These nodesrepresent decisions based on feature values.
# ####   Leaf Nodes: Terminal nodes that represent the final output, either a class label (in classification) or a continuous value (in regression)n)

# Practical Example: osition_Salaries dataset to predict the salary based on the level of position using Decision Tree Regression.

# ###  Decision Tree Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# load the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset


plt.scatter(dataset["Level"],dataset["Salary"],color='red')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


# Convert dataset into Row & Columns 
X = dataset.iloc[:,[1]].values
X

# Target Variable
Y = dataset.iloc[:,2].values
Y


# import the DT Libraries

from sklearn.tree import DecisionTreeRegressor

#model = DecisionTreeRegressor()
#model = DecisionTreeRegressor(max_depth=3)  --- Hyperparameter (Define the depth of Tree)
model = DecisionTreeRegressor(max_leaf_nodes=2)   # --- Hyperparameter


# Choose model based on preferece
# max_depth vs max_leaf_nodes
# Hyperparameters:
# max_depth: This parameter defines the maximum depth of the decision tree.
# By limiting the depth, you can prevent overfitting and control the complexity of the model
# max_leaf_nodes: This parameter sets the maximum number of leaf nodes in the decision tree.
# By limiting the number of leaf nodes, you can control the complexity of the tree.

# Fit the model
model.fit(X,Y)
DecisionTreeRegressor(max_leaf_nodes=2)


# Finding the model prediction score
model.score(X,Y)

#  R^2 score (coefficient of determination) is a statistical measure that indicates how well the regression model fits the data.
# It represents the proportion of the variance in the dependent variable (Y) that can be explained by the independent variable(s) (X) in the model.
# The R^2 score ranges from 0 to 1, where a score of
# 1 indicates a perfect fit (the model explains all the variance in the data),
# and a score of 0 indicates that the model does not explain any of the variance.
# A higher R^2 score indicates a better fit of the model to the data.
# In decision tree regression, the R^2 score is calculated based on the predictions made by the decision tree model compared to the actual values in the dataset.




#  manual prediction
model.predict(np.array([[6.5]]))
model.predict(np.array([[6.6]]))
model.predict(np.array([[6.3]]))
model.predict(np.array([[6.1]]))
model.predict(np.array([[6.0]]))
model.predict(np.array([[5.8]]))
model.predict(np.array([[5.9]]))
model.predict(np.array([[5.5]]))
model.predict(np.array([[5.3]]))

# in  this above models predictions are same 
# To overcome this we use  the following code for higher resolution and smoother curve
# The NumPy library that creates an array of evenly spaced values . 

X_grid = np.arange(1,10,0.01)

# arange(): This is a function provided by NumPy to 
#  generate an array with evenly spaced values within a specified range.
#  Start (1): The first value in the array. In this case, it starts at 1.
#  Stop (10): The endpoint of the range, which is exclusive. '
#             This means the generated array will not include 10.
#  (0.01): The difference between consecutive values in the array. Here, 
#           it specifies that each subsequent value will increase by 0.01.

X_grid

# reshape the data
X_grid = X_grid.reshape(len(X_grid),1)
X_grid

# check the dimension
X_grid.shape


# reshape the data 
# transforming X_grid into a two-dimensional array (also known as a matrix)

X_grid = X_grid.reshape(-1,1)
X_grid

# check the shape 
X_grid.shape


# plot the Distribution

plt.scatter(dataset['Level'],dataset['Salary'])
plt.plot(X_grid,model.predict(X_grid),color='purple')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


# import the DT library

from sklearn import tree
plt.figure(figsize=(8,10))
tree.plot_tree(model,filled=True)
#plt.savefig('rr.png')  save to png format to shaow DT
plt.show()

# sets up size of 8 inches in width and 10 inches in height.
# generates a visual representation of the decision tree stored in model.
# The filled=True argument indicates that the nodes in the tree should be color-coded based on their class labels or values.    
# displays the plotted decision tree.
