#!/usr/bin/env python
# coding: utf-8

"""      # Gradient Boosting Classification    """ 


# #### Gradient Boosting Classification is a powerful ensemble learning technique that builds a predictive model by combining multiple weak learners, typically decision trees. This method focuses on correcting the errors made by previous models, allowing it to achieve high accuracy in classification tasks.

# #### Key Concepts of Gradient Boosting Classification:
# 1. **Ensemble Learning**: Gradient Boosting combines multiple weak learners to create a strong predictive model. Each weak learner is trained sequentially, with each new model attempting to correct the errors of the previous ones.

# 2. **Weak Learners**: The weak learners are usually shallow decision trees (often referred to as stumps) that make predictions based on a small subset of features. These trees are simple and may not perform well individually, but when combined, they can produce a robust model.

# 3. **Gradient Descent**: The "gradient" in Gradient Boosting refers to the use of gradient descent optimization to minimize the loss function. The algorithm calculates the gradient of the loss function with respect to the predictions and uses this information to update the model iteratively.

# 4. **Loss Function**: The choice of loss function is crucial in Gradient Boosting. Common loss functions for classification tasks include log-loss (for binary classification) and multi-class log-loss (for multi-class classification). The algorithm aims to minimize this loss function during training.


"""  Praticle Implementation of Gradient Boosting Classification  """
# dataset: Social_Network_Ads.csv

# Load the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Load the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset



# Transforrm the dataset
X = dataset.iloc[:,2:4].values
X

# Target variable
Y = dataset.iloc[:,-1].values
Y


# import the sklearn library
from sklearn.model_selection import train_test_split

# split the dataset into train and testing set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0,test_size=0.2)
X_train
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
Y_test


# import  the sklearn ensemble fradient Classification library
from sklearn.ensemble import GradientBoostingClassifier

# create the model object
model = GradientBoostingClassifier(n_estimators=10)


# train the model 
model.fit(X_train,Y_train)

# predict the model with testing data
Y_pred = model.predict(X_test)
Y_pred


# Analyse the result with actual data vs predicted data

from sklearn.metrics import confusion_matrix,classification_report
print (confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

# Visualize the result
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Gradient Boosting Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualize the result for test data
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Gradient Boosting Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# #### Conclusion:

# Gradient Boosting Classification is a robust and effective technique for classification tasks, leveraging the power of ensemble learning and gradient descent optimization. By sequentially building weak learners that correct the errors of previous models, it can achieve high accuracy and generalization performance on various datasets.

