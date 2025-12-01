#!/usr/bin/env python
# coding: utf-8

"""  #  Decision Tree Classification     """

# Decision Tree (DT) is a supervised machine learning algorithm used for both classification and regression tasks. It works by recursively splitting the dataset into subsets based on the feature that provides the best separation of classes (in classification) or minimizes error (in regression).

# ### DT operate by splitting data into subsets based on feature values, forming a tree-like structure that aids in decision-making.

# ###  A decision tree consists of: 
# ###   Root Node:   The starting point that represents the entire dataset.
# ###   Internal Nodes: These nodesrepresent decisions based on feature values.
# ###   Leaf Nodes: Terminal nodes that represent the final output, either a class label (in classification) or a continuous value (in regression)

# Practical Example: Social Network Ads Dataset


# load the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')



# load the dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset


# plot the distribution

plt.scatter(dataset['Age'],dataset['EstimatedSalary'],c=dataset['Purchased'],cmap='rainbow')
plt.xlabel('AGE')
plt.ylabel('Est. Salary')
plt.show()


# convert the dataset into Rows and Columns
X = dataset.iloc[:,[2,3]].values
X
X.shape


y= dataset.iloc[:,-1].values
y
y.shape


# import the sklearn library 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Import the DT Classifier Library
from sklearn.tree import DecisionTreeClassifier

# prepare the model
model  = DecisionTreeClassifier(max_depth=2)
model

# fit the model
model.fit(X_train,y_train)

# evaluate the model
y_pred = model.predict(X_test)
y_pred
y_test

# accuracy of the model
# import Reports Library for Clarification And Analyse

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
from sklearn import tree


# plot the decision tree
plt.figure(figsize=(8,10))
tree.plot_tree(model,filled=True)
plt.savefig('rr.png')
plt.show()

# Feature Importance
importance = model.feature_importances_
importance

# gini = Impurities