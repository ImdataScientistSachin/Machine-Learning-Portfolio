#!/usr/bin/env python
# coding: utf-8

# # HW_Ex4_IceCream_UnitSalesPrediction

# In[ ]:





# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')


# In[ ]:





# In[6]:


# Load the dataset

dataset = pd.read_csv('Ice_cream_selling_data.csv')

dataset


# In[ ]:





# In[7]:


X = dataset.iloc[:, [0]].values


# In[29]:


X


# In[8]:


X.shape


# In[9]:


Y = dataset.iloc[:, [1]].values


# In[31]:


Y


# In[32]:


Y.shape


# In[ ]:





# In[33]:


# plot the Distribution

plt.scatter(X,Y)
plt.xlabel('Temp-->independent')
plt.ylabel('Ice Cream Sales ---> Dependent')
plt.show()


# In[ ]:





# In[14]:


# import library

from sklearn.linear_model import LinearRegression


# In[15]:


model = LinearRegression()


# In[16]:


model.fit(X,Y)


# In[17]:


# check the overall model score

model.score(X,Y)


# In[ ]:





# In[18]:


# Test the model Prediction using Manual data

test = np.array([[-6.316559]])

#([[6.5]])  Convert to 2-d array 


# In[41]:


test


# In[42]:


model.predict(test)


# In[50]:


# plot the value in distribution

plt.scatter(dataset['Temperature (°C)'],dataset['Ice Cream Sales (units)'])
plt.plot(X,model.predict(X),color='purple')
plt.xlabel('Temp')
plt.ylabel('Ice Cream Sales')
plt.show()


# In[ ]:





# In[51]:


# so in the above values we can clearly see the predicted values is not match up to the actual value
# so in that can we use Non- Linear regression (Polynomial Regression)


# In[ ]:





# ## Polynomial Regression

# In[71]:


# Import the librraries

from sklearn.preprocessing import PolynomialFeatures


# In[108]:


# Poly = PolynomialFeatures(degree=1)
Poly = PolynomialFeatures(degree=7)

# In the context of polynomial regression, the degree refers to the highest power of the independent variable in the polynomial equation. 
#It determines the level of complexity of the model.

# Degree 1 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋
# Degree 2 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋 + 𝛽2𝑋2 
# Degree 3 (Linear Regression):  𝑦=𝛽0+𝛽1𝑋 + 𝛽2𝑋2 + 𝛽3𝑋3

# As the degree increases, 
# the polynomial can capture more complex patterns in the data. However, 
# higher degrees can also lead to overfitting, where the model fits the training data too closely and may not generalize well to unseen data.


# In[ ]:





# In[110]:


X_poly = Poly.fit_transform(X)

# transform: Applies the learned transformation to the data.
# It takes the input data and generates polynomial features based on the specified degree.


# In[111]:


X_poly


# In[ ]:





# In[112]:


# prepare model

model1 = LinearRegression()


# In[113]:


model1.fit(X_poly,Y)


# In[114]:


model1.score(X_poly,Y)


# In[ ]:





# In[115]:


# prepare distribution

plt.scatter(dataset['Temperature (°C)'],dataset['Ice Cream Sales (units)'])
plt.plot(X,model1.predict(X_poly),color='green')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales (units)')
plt.show()


# In[ ]:





# In[116]:


# Testing the model

test1 = Poly.transform(test)

test1


# In[ ]:





# In[117]:


# pridict the model

model1.predict(test1)


# In[ ]:





# In[126]:


# manual testing 

raw_test2 = np.array([[3.704057]])

raw_test2


# In[ ]:





# In[127]:


test2 = Poly.transform(raw_test2)

test2


# In[128]:


model1.predict(test2)

# Preddicted value


# In[ ]:




