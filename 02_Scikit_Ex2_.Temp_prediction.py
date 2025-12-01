#!/usr/bin/env python
# coding: utf-8

""" # Scikit_Ex2_.Temp_prediction """


# prepare and load libraries 

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set_style('whitegrid')


Celsius = np.array([-40,-10,0,8,15,22,38])    # Independant



# prepare dataset (x)
Fahrenheit = np.array([-40,14,32,46,59,72,100])   # dependent


# Convert 1-d Array into 2-D array: apply only on independent variable
Celsius = Celsius.reshape(7,1)
Fahrenheit


# plot on distribution

plt.scatter(Celsius,Fahrenheit)
plt.ylabel('Fahrenheit')
plt.xlabel('Celsius')
plt.show()


# import LinearRegression function from scikit library 

from sklearn.linear_model import LinearRegression


# train model # model = function ,LinearRegression : class
model = LinearRegression()

# fit the model
model.fit(Celsius,Fahrenheit)

# model.fit(x,y)


# plot the intercept(b0) & coefficient(b1) 
# also don't have to add constant in scikit model

model.intercept_
model.coef_


# plot the Values

plt.scatter(Celsius,Fahrenheit)
plt.plot(Celsius,model.predict(Celsius),color='purple')   # in this we are apply indepent var values 
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.show()


# testing the model 
test = np.array([[1.0],[10]])    # convert series data into  array
test

# predict the values
model.predict(test)
# predict function always take 2-D array
# check for 1 value
model.predict([[100]])

# check the score of the model  (r-squared)
model.score(Celsius,Fahrenheit)
