#!/usr/bin/env python
# coding: utf-8

"""  # ## linear_Regression_Assumptions_Task  """
# ### 1. Autocorrelation
# ### 2. Multivariate Normality
# ### 3. Linear Relationship
# ### 4. Homoscedasticity

# Load Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.datasets import make_regression


x,y = make_regression(n_features=3,n_samples=500,noise=0.6 )
# dataset =make_regression(n_features=3,n_samples=500,noise=0.6 )
#  n_features=3  : Independent Variable
# n_samples=500  : dataset :Columns
# noise=0.6  : Scatter data        
x
x.shape


Dfx= pd.DataFrame(x,columns=['X1','X2','X3'])
Dfx
Dfy = pd.DataFrame(y,columns=['Asumptiion'])
Dfy


import statsmodels.api as sm

# Add constant
const = sm.add_constant(Dfx)
const

# prepare model
model = sm.OLS(Dfy,const).fit()
model.summary()
# OLS : Ordinary Least Square

# 1.Test Method:  Autocorrelation  Durbin-Watson:1.792
 
#---> our data has positive  Autocorealion

# 2. Test Method:-- Multivariate Normallity

sns.distplot(Dfx['X1'])
plt.show()

Dfx.describe()
# gives the detail infoormation about the plot
sns.distplot(Dfx['X2'])
plt.show()

sns.distplot(Dfx['X3'])
plt.show()

sns.distplot(Dfy['Asumptiion'])
plt.show()

Dfy.describe() 
# gives the detail infoormation about the plot


# 2nd Test method QQ - Test Method
Q_Q_plot = sm.qqplot(model.resid,fit=True)

plt.title('Q-Q plot of Residuals')
plt.show()
# if the plot follow the straight line then our data is normality


# 3. Test Method:#  Linear Relationship

plt.scatter(Dfx['X1'],Dfy['Asumptiion'],color='red')
plt.xlabel('X1')
plt.ylabel('Asumptiion')
plt.show()

# Scatter plot
plt.scatter(Dfx['X2'],Dfy['Asumptiion'],color='purple')
plt.xlabel('X2')
plt.ylabel('Asumptiion')
plt.show()

# Scatter plot
plt.scatter(Dfx['X3'],Dfy['Asumptiion'],color='Green')
plt.xlabel('X3')
plt.ylabel('Asumptiion')
plt.show()



# 4 .  Test Method:  Homoscedasticity
import statsmodels.stats.api as sms



# function to test  Homoscedasticity
sms.het_goldfeldquandt(model.resid,model.model.exog)


# second value is P value
# H0: your data is Homoscedasticity
# 0.95 > 0.05  ---> Accept -----> means our data is Homoscedasticity
# 0.01 < 0.05  -----> Reject -----> means our data is Heteroscedasticity


# 4.  Test Method:   Multicolinearity

# find correlations
Dfx

#  find the Coefficient-Correlation

Dfx.corr()
# VIF method

from statsmodels.stats.outliers_influence import variance_inflation_factor
Dfx.values

# test VIF function
vif = [variance_inflation_factor(Dfx.values,i) for i in range(3)]
vif
vif_df = pd.DataFrame()
vif_df['VIF'] = vif
vif_df['Features'] = Dfx.columns
vif_df

# result Lower then 5 ....Acceptable
# if VIF > 5 -----> Multicolinearity exists
# if VIF < 5 -----> No Multicolinearity exists
# ###  Exercise Completed  ###
