#!/usr/bin/env python
# coding: utf-8

"""   # Naive Bays Classification      """

# #### Naive Bayes classification is a fundamental technique in machine learning that utilizes Bayes' Theorem to perform classification tasks.especially those involving high-dimensional data like text.

# ##### Probabilistic Approach: Naive Bayes looks at the probability of each category (or class) given the features of the data. 

# #### Types of Naive Bayes:
# #### Gaussian Naive Bayes: Used for continuous data (like height or weight).
# #### Multinomial Naive Bayes: Best for count data (like word counts in texts) .
# #### Bernoulli Naive Bayes: Used for binary data (like whether a word appears or not)..


"""   Practicle Implementation   """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# Load the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset



#  transform the dataset
X = dataset.iloc[:,[2,3]].values
X


# target Variable
Y = dataset.iloc[:,-1].values
Y


# import the sklearn Library

from sklearn.model_selection import train_test_split

# split the dataset
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape



# import the NBC packages

from sklearn.naive_bayes import GaussianNB


# create model
model = GaussianNB()

# train the model
model.fit(X_train,y_train)



y_pred = model.predict(X_test)
y_pred
y_test

# import the analyse package for analyse
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_pred,y_test))


# analyse the overaall  model score
print(classification_report(y_test,y_pred))