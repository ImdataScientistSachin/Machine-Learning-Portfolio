#!/usr/bin/env python
# coding: utf-8

"""    # Gradient Boosting Regression     """


# #### Gradient Boosting Regression is an advanced ensemble learning technique that builds predictive models by combining multiple weak learners, typically decision trees. This method enhances model accuracy by sequentially adding models that correct the errors of previous iterations, making it particularly effective for regression tasks.


"""  Practicle of Gradient Boosting Regression  """


# import the Libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# Load the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset



# prepare the distribution 

plt.scatter(dataset['Level'],dataset['Salary'])
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()



# convert the dataset in 2 D array
X = dataset.iloc[:,[1]].values
X

# convert the dataset in 2 D array
Y = dataset.iloc[:,[-1]].values
Y



# import the  Gradient Booting Library 
from sklearn.ensemble import GradientBoostingRegressor

# create the model
model = GradientBoostingRegressor(n_estimators=16)

# train model
model.fit(X,Y)

# check the score of model
model.score(X,Y)


# maually Prediction
model.predict(np.array([[6.5]]))

# Visualising the Gradient Boosting Regression results with higher resolution and smoother curve
# Convert the value into subsets

X_grid = np.arange(1,10,0.01)
X_grid



# reshape the dataset
X_grid = X_grid.reshape(-1,1)
X_grid


# plot the graph
# Visualising the Gradient Boosting Regression results

plt.scatter(dataset['Level'],dataset['Salary'])
plt.plot(X_grid,model.predict(X_grid),color='black')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
