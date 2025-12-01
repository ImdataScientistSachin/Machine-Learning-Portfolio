#!/usr/bin/env python
# coding: utf-8

"""  # Support-Vector-Machine-Regression    """


# #### Key Features : 
# #### Epsilon-Insensitive Loss Function: SVR uses an epsilon-insensitive loss function, which means it ignores errors that fall within a specified margin (epsilon). This allows the model to be robust to small deviations and focuses on fitting the data points that are outside this margin.

# #### Support Vectors: Just like in SVM, SVR identifies support vectors, which are the data points that are closest to the predicted function (hyperplane). These points are critical because they influence the position and orientation of the hyperplane.

# #### Kernel Trick: SVR can handle non-linear relationships between input features and output by applying kernel functions. 

# Common kernels include:
# #### Linear Kernel: For linear relationships.
# #### Polynomial Kernel: For polynomial relationships.
# #### Radial Basis Function (RBF) Kernel: For more complex, non-linear relationships.


""" Practicle Implementation """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# load the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset


plt.scatter(dataset['Level'],dataset['Salary'])
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()



# transform the dataset into Rows and Cloumn
X = dataset.iloc[:,[1]].values
X

# target variable
Y = dataset.iloc[:,2].values
Y



# import the SVR packages
from sklearn.svm import SVR


# linear,sigmoid,poly,rbf
model = SVR(kernel='poly',degree=8)

# 'poly' indicates that a polynomial kernel will be employed. 
# This kernel allows the model to learn non-linear relationships by transforming the input space into a higher-dimensional space using polynomial functions.

# degree=8: This parameter sets the degree of the polynomial kernel. 
# A degree of 8 means that the polynomial function used in the kernel will be of the 8th order.


# traine the model 
model.fit(X,Y)
model.score(X,Y)


plt.scatter(dataset['Level'],dataset['Salary'])
plt.plot(X,model.predict(X),color='red')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# manually prediction
model.predict(np.array([[6.5]]))
