#!/usr/bin/env python
# coding: utf-8

# # Pipelines in ML .

# ### What Are Pipelines in Machine Learning?
# #### A machine learning (ML) pipeline is an organized, automated sequence of steps that takes raw data through all necessary stages—such as preprocessing, feature engineering, model training, evaluation, and deployment—to deliver a working ML model ready for predictions. Each step’s output becomes the input for the next, ensuring a smooth and reproducible workflow.
# 
# ### Key Components of an ML Pipeline
# ##### Data Collection: Gathering data from various sources like databases, APIs, or files.
# 
#  ##### Data Preprocessing: Cleaning and structuring raw data by handling missing values, removing duplicates, normalizing, encoding, and splitting into training/testing sets.
# 
# ##### Feature Engineering: Extracting or creating relevant features from the data to improve model performance, such as using PCA, scaling, or selecting important variables.
# 
# ##### Model Training: Selecting and training an ML algorithm on the processed data.
# 
# ##### Model Evaluation: Assessing the model’s performance using metrics like accuracy, precision, recall, or cross-validation.
# 

# In[ ]:





# # dataset without using pipeline vs using pipeline 

# ### Part 1) dataset without using pipeline

# In[ ]:





# In[1]:


# import the libraries


# In[2]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


# In[3]:


df = pd.read_csv('train.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[ ]:





# In[6]:


# drop unnecessary column 

df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace = True)

# inplace = True 
# The inplace=True parameter in Pandas operations like df.drop() means the operation is applied directly to the original DataFrame instead of returning a modified copy.
# Default behavior: Without inplace=True (or with inplace=False), Pandas creates and returns a new DataFrame with the changes, leaving the original unchanged .


# In[25]:


# New datasrt

df.head()


# In[26]:


df.shape


# 
# ### prepare train/test/split

# In[33]:


X_train,X_test,y_train,y_test= train_test_split(df.drop(columns=['Survived']),
                                               df['Survived'],
                                               test_size=0.2,
                                               random_state=42)


# In[34]:


print(X_train.shape)

X_train.head(3)


# In[35]:


print(y_train.shape)

y_train.head(3)


# In[ ]:





# #### B) Find Missing Values

# In[36]:


df.isnull().sum()


# In[37]:


# Age &  Embarkrd have missing values


# #### C) Fill the missing values

# In[38]:


# Applying imputation (fill missing value with Mean and  most frequent)

si_age = SimpleImputer()         # Default strategy='mean' for numerical Age column
si_embarked = SimpleImputer(strategy='most_frequent') # For categorical Embarked column 


# Handle missing values in TRAINING data
X_train_age = si_age.fit_transform(X_train[['Age']])      # Fit imputer and transform Age column 
X_train_embarked = si_embarked.fit_transform(X_train[['Embarked']])


# Apply learned imputation to TEST data
X_test_age = si_age.transform(X_test[['Age']])      # Use same mean from training for Age  
X_test_embarked = si_embarked.transform(X_test[['Embarked']])  # Use same frequent category from training   


# In[39]:


X_test_age


# In[40]:


X_train_embarked


# In[ ]:





# #### Apply one hot encoding on Sex and Embarked (categorical values)

# In[41]:


ohe_sex = OneHotEncoder(sparse_output=False,handle_unknown='ignore')  # For Sex column (categories: male/female) 
ohe_embarked = OneHotEncoder(sparse_output=False,handle_unknown='ignore')  # For Embarked column (categories: C/Q/S) 

# Encode TRAINING data
X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])  # Learn categories and transform Sex
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked)  # Learn categories and transform Embarked

# Encode TEST data (using learned categories from training)
X_test_sex = ohe_sex.transform(X_test[['Sex']])  # Transform Sex using training categories
X_test_embarked = ohe_embarked.transform(X_test_embarked)  # Transform Embarked using training categories


# Parameters:
# sparse=False: Returns a dense array (matrix) instead of sparse format
# handle_unknown='ignore': Prevents errors if test data contains unseen categories (make default : 0)


# In[42]:


X_train_sex 


# In[43]:


X_train_embarked

# [C,S,Q]


# In[44]:


# we have create 3 new arrays sex, Embarked, Age

X_train.head(2)


# In[45]:


# create a new array fro remaing column 

X_train_rem = X_train.drop(columns=['Sex','Age','Embarked'])

X_train_rem


# In[50]:


X_test_rem = X_test.drop(columns=['Sex','Age','Embarked'])

print(X_test_rem.shape)

X_test_rem


# In[ ]:





# In[ ]:





# In[51]:


# check shape 

print(X_train_rem.shape)
print(X_train_age.shape)
print(X_train_sex.shape)
print(X_train_embarked.shape)


# In[52]:


print(X_test_rem.shape)
print(X_test_age.shape)
print(X_test_sex.shape)
print(X_test_embarked.shape)


# In[ ]:





# ### D) Transform Column

# In[53]:


# Combine preprocessed components into final arrays

X_train_transformed = np.concatenate(
    (X_train_rem, X_train_age, X_train_sex, X_train_embarked),
    axis=1  # Merge columns horizontally
)

X_test_transformed = np.concatenate(
    (X_test_rem, X_test_age, X_test_sex, X_test_embarked),
    axis=1  # Match training data structure
)


# In[55]:


# check shape

X_test_transformed.shape


# In[ ]:





# ### train model

# In[56]:


# use DicisionTree 

clf = DecisionTreeClassifier()

clf.fit(X_train_transformed,y_train)


# ### check prediction
# 

# In[58]:


y_pred = clf.predict(X_test_transformed)

y_pred


# ### check prediction score 
# 
# 

# In[60]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)


# In[61]:


score


# ## Deployment

# In[63]:


import pickle


# In[66]:


pickle.dump(ohe_sex,open('models/ohe_sex.pkl','wb'))  # Save OneHotEncoder for 'Sex' column to disk
pickle.dump(ohe_embarked,open('models/ohe_embarked.pkl','wb'))  # Save OneHotEncoder for 'Embarked' column to disk 
pickle.dump(clf,open('models/clf.pkl','wb'))  # Save trained classifier model to disk


# this file save in models folder

# The 'wb' mode in Python's open() function stands for "write binary." 
# It is used when you want to write binary data (like bytes, images, or serialized objects) to a file, rather than plain text. 


# In[ ]:




