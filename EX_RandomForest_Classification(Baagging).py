#!/usr/bin/env python
# coding: utf-8

"""    # Random Forest_Classification(Baagging)   """

# #### Ensemble Learning: Random Forest combines multiple decision trees to improve model accuracy. Each tree is trained on a random subset of the data, which helps to ensure diversity among the trees.

# #### Bagging (Bootstrap Aggregating): Random Forest uses bagging to create multiple subsets of the training data by sampling with replacement. Each decision tree is trained on a different subset, which helps to reduce overfitting and improve generalization.

# #### Random Forest is used for both classification and regression tasks. In classification, it predicts the class label based on the majority vote from individual trees.

# Practical Example: Social_Network_Ads dataset to predict whether a user will purchase a product based on their age and estimated salary using Random Forest Classification.


# load Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# prepare the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset



# convert the dataset into Rows & Columns
X = dataset.iloc[:,2:4].values
X

# Target Variable
Y = dataset.iloc[:,-1].values
Y


# import the sklearn library
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape


# Import the Random Forest classifier Lbrary

from sklearn.ensemble import RandomForestClassifier

# create the model
model = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=3)

# n_estimators: The number of trees in the forest.
# max_depth: The maximum depth of each tree (optional).
# min_samples_split: The minimum number of samples required to split an internal node (optional).
# random_state: Controls the randomness of the estimator (optional).



# train Model
model.fit(X_train,Y_train)

# Predict the model
Y_pred = model.predict(X_test)
Y_pred

# Actual Test data
Y_test


# Compaire the result

from sklearn.metrics import confusion_matrix,classification_report


print(confusion_matrix(Y_test,Y_pred))

# Compare Actual datat vs Predicted data
print(classification_report(Y_test,Y_pred))
# Precision: The ratio of true positive predictions to the total predicted positives.
# Recall: The ratio of true positive predictions to the total actual positives.
# F1-score: The harmonic mean of precision and recall, providing a single metric that balances both.
# Accuracy: The overall accuracy of the model, representing the proportion of correct predictions.
# In this example, the Random Forest Classifier is trained on the Social_Network_Ads dataset to predict whether a user will purchase a product based on their age and estimated salary. The model's performance is evaluated using a confusion matrix and classification report.


# Visualize the result
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
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')   
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In this visualization, the decision boundary created by the Random Forest Classifier is shown along with the test data points. The red and green regions represent the predicted classes, while the points represent the actual test data. This helps to visualize how well the model separates the two classes based on age and estimated salary.

# Note: You can adjust the hyperparameters of the Random Forest Classifier (e.g., n_estimators, max_depth) to optimize the model's performance based on your specific dataset and requirements.

# In this example, the Random Forest Classifier is trained on the Social_Network_Ads dataset to predict whether a user will purchase a product based on their age and estimated salary. The model's performance is evaluated using a confusion matrix and classification report.    

# #### Conclusion: Random Forest Classification with Bagging is a powerful ensemble learning technique that combines multiple decision trees to improve model accuracy and generalization. By using bagging, it reduces overfitting and enhances the robustness of the model, making it suitable for various classification tasks.