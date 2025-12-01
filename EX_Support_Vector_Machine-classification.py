#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine classification

# #### Support Vector Machine (SVM) classification is a powerful supervised learning algorithm used primarily for binary classification tasks. Here’s a concise overview of its key concepts, workings, and applications.

# ##### SVM is designed to find the best decision boundary (hyperplane) that separates data points of different classes in a high-dimensional space. 

# ##### Key Concepts : 
#   
# #####  Hyperplane: A hyperplane is a flat affine subspace that divides the data into two classes. In a two-dimensional space, it is simply a line; in three dimensions, it is a plane.  
# ##### Support Vectors: These are the data points that are closest to the hyperplane and influence its position and orientation. The SVM algorithm focuses on these points for constructing the decision boundar .
# #####  Margin: The margin is the distance between the hyperplane and the nearest support vectors from either class. SVM aims to maximize this margin, which helps improve classification accurac.

# In[ ]:





# ###### How SVM Works
# ######  Training Phase: During training, SVM identifies the optimal hyperplane that maximizes the margin between classes using support vectors.
# ######    Prediction Phase: For new data points, SVM determines which side of the hyperplane they fall on to classify them into one of the categories.

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[4]:


dataset = pd.read_csv('Social_Network_Ads.csv')


dataset


# In[6]:


# plot the distribution

plt.scatter(dataset['Age'],dataset['EstimatedSalary'],c=dataset['Purchased'],cmap='rainbow')
plt.xlabel('Age')
plt.ylabel('Est. Salary')
plt.show()


# In[9]:


# transform the dataset in Rows and Cloumn

X = dataset.iloc[:,[2,3]].values

X


# In[10]:


Y = dataset.iloc[:,-1].values
Y


# In[ ]:





# In[12]:


# import the sklearn libraries

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[ ]:





# In[13]:


# import the SVC packages

from sklearn.svm import SVC


# In[16]:


# prepare the model

model = SVC(kernel='rbf')

#### kernel='rbf': This parameter specifies the type of kernel function to be used in the SVC model. 
#  The RBF kernel is a popular choice for handling non-linear classification problems.
# It maps input features into a higher-dimensional space where a linear separation is possible.


# In[ ]:





# In[17]:


model.fit(X_train,y_train)


# In[19]:


# model prediction

y_pred = model.predict(X_test)

y_pred


# In[20]:


# Analyse the prediction with actual data

from sklearn.metrics import confusion_matrix,classification_report


# In[21]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:





# In[22]:


# analyse the overall model score 

print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:




