#!/usr/bin/env python
# coding: utf-8

"""  # Scikit_Ex1_Negative_Dataset """

# prepare and load libraries 

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set_style('whitegrid')
import pandas as pd


# prepare dataset (x)
x = np.array([-1,0,1,2,3,4])

# Convert 1-d Array into 2-D array: apply only on independent variable
x = x.reshape(6,1)
x

# prepare dataset (y)
y = np.array([-3,-1,1,3,5,7])
y


# plot on distribution
plt.scatter(x,y)
plt.xlabel('Independent')
plt.ylabel('Dependent')
plt.show()


# import LinearRegression function from scikit library 
from sklearn.linear_model import LinearRegression
# train model # model = function ,LinearRegression : class

model = LinearRegression()
model.fit(x,y)

# model.fit(x,y)

# plot the intercept(b0) & coefficient(b1) 
# also don't have to add constant in scikit model

model.intercept_
model.coef_


# plot the Values

plt.scatter(x,y)
plt.plot(x,model.predict(x),color='green')   # in this we are apply indepent var values 
plt.xlabel('Independent')
plt.ylabel('Dependent')
plt.show()

# testing the model 
test = np.array([[1.0],[5]])    # convert series data into  array
test

# predict the values
model.predict(test)

# check the score of the model  (r-squared)
model.score(x,y)
