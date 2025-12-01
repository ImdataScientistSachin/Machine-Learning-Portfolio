#!/usr/bin/env python
# coding: utf-8

""" # K-Nearest Neighbour Classification  """


# #### K-Nearest Neighbors (KNN) is a simple yet powerful machine learning algorithm used for both classification and regression tasks.

# #### KNN is a non-parametric and supervised learning algorithm. 
# #### It operates on the principle that similar data points are likely to be found close to each other in the feature space.

# ##### How KNN Works
# ##### Distance Calculation: When a new data point needs to be classified, KNN calculates the distance between this point and all existing points in the dataset. Common distance metrics include:

#    Euclidean Distance 
#    Manhattan Distance
#    Minkowski Distance

# #####   Identifying Neighbors: After calculating distances, KNN identifies the 'k' closest points (neighbors) to the new data point. The value of 'k' is a crucial parameter that determines how many neighbors will be considered during classification.

# ##### Voting Mechanism: For classification tasks, KNN assigns the class label that is most common among the 'k' neighbors (majority voting). In regression tasks, it predicts the average of the values of these neighborsbors

# #### Choosing the Right 'k' Value
# #### The choice of 'k' significantly impacts the performance of the KNN algorithm. A small 'k' value can lead to overfitting, while a large 'k' value may result in underfitting. Common practices for selecting 'k' include:

#   Cross-Validation: Using techniques like k-fold cross-validation to evaluate different 'k' values and select the one that yields the best performance.


""" Praticle Implementation of K-Nearest Neighbour Classification using Python's sklearn library  """


# Loading Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import pandas as pd



# import the Iris dataset from sklearn lirary

from sklearn.datasets import load_iris


Iris = load_iris()
Iris


X = Iris.data
X

# target variable
Y = Iris.target
Y



# load the model 
from sklearn.model_selection import train_test_split

# split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# import the KNN library
from sklearn.neighbors import KNeighborsClassifier




# prepare model
model = KNeighborsClassifier(n_neighbors=5,p=2,weights='uniform')

# n_neighbors=5: This parameter specifies the number of nearest neighbors to consider when making a classification decision. 
# p=2: This parameter defines the distance metric used to calculate the distance between data points
# weights='uniform': This parameter determines how the contributions of each neighbor are weighted when making a prediction


# train model 
model.fit(X_train,Y_train)


#model Prediction 
y_pred = model.predict(X_test)
y_pred

# actual test values 
Y_test



# analysed the prediction vs Actual data

from sklearn.metrics import confusion_matrix,classification_report


# print Confusion matrix to analyse the prediction
print(confusion_matrix(Y_test,y_pred))


# analyse the overaall performance of model
print(classification_report(Y_test,y_pred))

# Accuracy of the model
accuracy = model.score(X_test,Y_test)
print("Accuracy of the model:", accuracy)

# Accuracy of the model: 1.0
# The model achieved an accuracy of 100% on the test dataset, indicating that it correctly classified all test instances.
# In real-world scenarios, achieving perfect accuracy is rare, and it's essential to evaluate the model's performance using various metrics and validation techniques.
# In this implementation, we loaded the Iris dataset, split it into training and testing sets, trained a K-Nearest Neighbors classifier, and evaluated its performance using a confusion matrix and classification report. The model achieved perfect accuracy on the test set.

