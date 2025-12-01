#!/usr/bin/env python
# coding: utf-8

"""  #  Radius Neighbors Classification  """ 

# #### ** Classification based on Radius 

#### ** What is Radius Neighbors Classification? **

# ##### Radius Neighbors Classification is a machine learning algorithm that extends the k-nearest neighbors (k-NN) approach by predicting class labels based on all training examples within a specified radius, rather than a fixed number of nearest neighbors. This method is particularly useful in scenarios where data is sparse or when the density of data points varies significantly across the feature space.


""" Practicle Example of Radius Neighbors Classification  """


# load the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# create an array of data points
X = np.array([[0,1],[0.5,1],[3,1],[3,2],[1.3,0.8],[2.5,2.5],[2.4,2.6]])
X

# create the target variable
Y = np.array([0,0,1,1,0,1,1])
Y



# Plot the distribution 

plt.scatter(X[:,0],X[:,1],c=Y,cmap='rainbow')
plt.show()

#  X[:,0] and X[:,1]: These represent the data for the x-axis and y-axis . 
#  [:,0] extracts all rows from the first column (x-coordinates).
#  X[:,1] extracts all rows from the second column (y-coordinates) .
#  c=Y: This argument specifies the color of each point in the scatter plot .



# import the Radius classifier  library
from sklearn.neighbors import RadiusNeighborsClassifier

# create the model
model = RadiusNeighborsClassifier(radius=1.0,outlier_label=2)

# radius=1.0: defines the radius within which to search for neighbors when classifying a data point 

# utlier_label=2: specifies the label that should be assigned to outlier samples—those, 
# that do not have any neighbors within the defined radius.

model

# fit the model
model.fit(X,Y)

# predict a new value
model.predict(np.array([[30,20]]))

# np.array([[30, 20]]):This creates a 2D NumPy array containing a single sample with two features (30 and 20).

# 30 could represent the value of the first feature (e.g., age, size, etc.).
# 20 could represent the value of the second feature (e.g., another measurement relevant to the prediction).

# visualize the radius
plt.scatter(X[:,0],X[:,1],c=Y,cmap='rainbow')
circle = plt.Circle((10,20),1,color='r',fill=False)
plt.gca().add_patch(circle)
plt.show()

# Here, we visualize the radius around the point (10, 20) with a radius of 1. The red circle represents the area within which the model would look for neighbors to classify the point (10, 20).
# However, since there are no points within this radius, the model would classify (10, 20) as an outlier and assign it the label 2.
# In summary, Radius Neighbors Classification is a useful technique for classifying data points based on their proximity to other points within a specified radius, making it particularly effective in scenarios with varying data densities.
# The scatter plot shows the original data points colored by their class labels, and the red circle indicates the radius around the point (10, 20).
