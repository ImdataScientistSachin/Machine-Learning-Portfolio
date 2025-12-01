#!/usr/bin/env python
# coding: utf-8

# ## Recommendation Systems: Build recommendation systems for users based on app features and ratings. ( org)

# ### Step - 1. Setup and Data Preparation

# In[ ]:







# In[37]:


# import the libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[38]:


# 1. Data Loading and Initial Exploration

dataset = pd.read_csv('googleplaystore.csv')
print("Original Dataset Shape:", dataset.shape)
print(dataset.head())


# In[ ]:





# ### Step - 2. Data Cleaning and Preprocessing

# In[39]:


backup_dataset= dataset.copy()  # create a backup copy of dataset


# In[40]:


dataset.info()


# In[41]:


dataset['Category']


# In[42]:


# Convert 'Installs' to string first to handle potential non-string entries

dataset['Installs'] = dataset['Installs'].astype(str).str.replace(r'[+,]', '', regex=True)


# In[43]:


dataset = dataset[dataset['Installs'].str.isnumeric()]  # Keep only numeric values
dataset['Installs'] = dataset['Installs'].astype(int)


# In[44]:


# Clean 'Price' column
dataset['Price'] = dataset['Price'].str.replace('$', '', regex=False).astype(float)


# In[45]:


# Convert 'Last Updated' to datetime
dataset['Last Updated'] = pd.to_datetime(dataset['Last Updated'])


# In[46]:


# Convert 'Reviews' to numeric
dataset['Reviews'] = pd.to_numeric(dataset['Reviews'], errors='coerce')


# In[47]:


# Handle 'Size'

def convert_size_to_mb(size):
    if isinstance(size, str):
        if 'M' in size:
            return float(size.replace('M', ''))
        elif 'k' in size:
            return float(size.replace('k', '')) / 1024
        elif 'Varies with device' in size:
            return np.nan  # Or a suitable placeholder
    return np.nan


# In[48]:


dataset['Size'] = dataset['Size'].apply(convert_size_to_mb)
dataset['Size'] = pd.to_numeric(dataset['Size'], errors='coerce') # Handle any conversion errors


# In[49]:


# Impute missing 'Size' values using median for each category

dataset['Size'] = dataset.groupby('Category')['Size'].transform(lambda x: x.fillna(x.median()))


# In[50]:


# Outlier Handling (Reviews) - BEFORE SPLITTING

Q1 = dataset['Reviews'].quantile(0.25)
Q3 = dataset['Reviews'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
dataset['Reviews'] = np.clip(dataset['Reviews'], lower_bound, upper_bound)


# In[ ]:





# ### Step- 3. Feature Engineering

# In[51]:


# App Age (in days)
dataset['App Age'] = (pd.to_datetime('2018-12-31') - dataset['Last Updated']).dt.days  # Assuming analysis date is end of 2018
dataset['App Age'] = dataset['App Age'].fillna(dataset['App Age'].median()) #handle any missing values after calculation

dataset['Features'] = dataset['Category'] + ' ' + dataset['Genres'] + ' ' + dataset['App']  #Includes App Name


# #### Step - 4 Split Data (Crucially, *before* imputation and encoding)

# In[52]:


# Remove rows with NaN in 'Rating' BEFORE the split
dataset = dataset.dropna(subset=['Rating'])


# In[53]:


# Split Data (Crucially, *before* imputation and encoding)

X = dataset.drop('Rating', axis=1)
y = dataset['Rating']


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# ### step - 5. Define Numerical and Categorical Features (AFTER SPLIT)

# In[55]:


numerical_features = ['Reviews', 'Size', 'Installs', 'Price', 'App Age']
categorical_features = ['Type', 'Content Rating']


# #### Step - 6. Imputation (AFTER SPLIT)

# In[56]:


print("Starting Imputation...")

imputer_numerical = SimpleImputer(strategy='median')
# Use median for robustness
print("SimpleImputer created.")

# Impute missing values separately for each column
print("Starting imputation loop...")


# In[57]:


# Impute missing values separately for each column

for col in numerical_features:
    print(f"Imputing column: {col}")
    X_train[col] = X_train[[col]].fillna(X_train[[col]].median())
    X_test[col] = X_test[[col]].fillna(X_test[[col]].median())
    print(f"Column {col} imputed.")
print("Imputation complete.")


# #### step -  7. Scaling (AFTER IMPUTATION)

# In[58]:


scaler = MinMaxScaler()
numerical_scaled_train = scaler.fit_transform(X_train[numerical_features])
numerical_scaled_test = scaler.transform(X_test[numerical_features])


# #### Step - 8. Encoding (AFTER SPLIT)

# In[59]:


encoder = OneHotEncoder(handle_unknown='ignore')
encoded_data_train = encoder.fit_transform(X_train[categorical_features])
encoded_data_test = encoder.transform(X_test[categorical_features])


# In[60]:


# Export the file for further use 

# dataset.to_csv('sorted_dataset.csv', index=False)


# In[61]:


dataset.head()


# In[ ]:





# ### step - 9. TF-IDF Vectorization:

# In[62]:


# --- TF-IDF Vectorization ---

tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_features=1000) # reduced ngram_range, added max_features
tfidf_matrix_train = tfidf_vectorizer.fit_transform(X_train['Features']).toarray()
tfidf_matrix_test = tfidf_vectorizer.transform(X_test['Features']).toarray()


# ##### Explanation:
# 
#   ##### TfidfVectorizer: This converts the text in the "Features" column into a numerical representation that machine learning models can understand. TF-IDF (Term Frequency-Inverse Document Frequency) weighs words based on their importance within each app's description and across the entire dataset.
# 
# ##### stop_words='english': This removes common English words (like "the", "a", "is") that don't contribute much to the meaning and can clutter the analysis.
# 
# ##### tfidf_matrix.shape: The output will show you the dimensions of the matrix. 

# ### Step -10. Sparse Matrix Conversion

# In[64]:


import gc  # Import garbage collection

numerical_scaled_train_sparse = csr_matrix(numerical_scaled_train, dtype=np.float32)  # Added dtype
gc.collect()
numerical_scaled_test_sparse = csr_matrix(numerical_scaled_test, dtype=np.float32) # Added dtype
gc.collect()
encoded_data_train_sparse = csr_matrix(encoded_data_train, dtype=np.float32) # Added dtype
gc.collect()
encoded_data_test_sparse = csr_matrix(encoded_data_test, dtype=np.float32) # Added dtype
gc.collect()


# #### Step - 11 Feature Combination (Sparse)

# In[65]:


combined_features_train = hstack([tfidf_matrix_train, numerical_scaled_train_sparse, encoded_data_train_sparse])
combined_features_test = hstack([tfidf_matrix_test, numerical_scaled_test_sparse, encoded_data_test_sparse])

print("Combined features (train) shape:", combined_features_train.shape)
print("Combined features (test) shape:", combined_features_test.shape)


# In[ ]:





# ###  Step - 12 - Prepare model - HYPERPARAMETER TUNING

# In[66]:


param_grid = {
    'n_estimators': [100, 200], # Reduce n_estimators
    'max_depth': [None, 10], #Reduce max_depth
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# In[67]:


# Step -13 - train the model (USING BEST MODEL)

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3,  # 3-fold cross-validation
                           scoring='neg_mean_squared_error',  # or 'r2'
                           verbose=2,
                           n_jobs=-1)  


# In[68]:


# Use all available cores

grid_search.fit(combined_features_train, y_train)

best_model = grid_search.best_estimator_  # Use the best model


# In[69]:


# Step -13 - train the model (USING BEST MODEL)
# best_model.fit(combined_features_train, y_train) 
# No need to fit again, GridSearchCV already did


# In[ ]:





# ### Step - 14 # --- Make Predictions ---

# In[70]:


y_pred = best_model.predict(combined_features_test)


# ### Step - 15. Evaluation

# In[81]:


# --- Evaluate the Model ---

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")


# In[ ]:





# #### Step -16. Recommendation Function

# In[82]:


def recommend_apps(app_name, num_recommendations=5):
    """
    Recommends similar apps based on feature similarity using the trained model.
    """
    try:
        # Get the features of the target app
        #app_index = X[X['App'] == app_name].index[0]

        # Find the app in the *original* dataframe
        app_data = dataset[dataset['App'] == app_name].iloc[0].to_frame().T # Get the row of the given app, from the parent dataset
        app_data = app_data.drop('Rating', axis=1) # Remove Rating from this entry for processing.

        # Impute and encode features
        for col in numerical_features: #Impute numerical features
            app_data[col] = app_data[col].fillna(X_train[col].median())
        numerical_scaled_app = scaler.transform(app_data[numerical_features]) #Scale numerical features
        encoded_data_app = encoder.transform(app_data[categorical_features]) #Encode categorical features
        tfidf_matrix_app = tfidf_vectorizer.transform(app_data['Features']) #Create the TFIDF matrix

        #Ensure matrixes are sparse
        numerical_scaled_app_sparse = csr_matrix(numerical_scaled_app)
        encoded_data_app_sparse = csr_matrix(encoded_data_app)
        combined_features_app = hstack([tfidf_matrix_app, numerical_scaled_app_sparse, encoded_data_app_sparse])

        # Make Prediction
        predicted_rating = best_model.predict(combined_features_app)[0]
        print(f"Predicted rating for the app: {predicted_rating:.2f}")

        #Compute Similarity
        similarity_scores = cosine_similarity(combined_features_app, combined_features_train)
        similar_app_indices = similarity_scores.argsort()[0][-(num_recommendations + 1):-1][::-1]

        # Print Results
        recommended_apps = X_train.iloc[similar_app_indices]['App'].values
        print(f"Recommended apps similar to {app_name}:")
        for app in recommended_apps:
            print(app)

    except IndexError:
        print(f"App '{app_name}' not found in the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")


# #### step 17 - Example usage:
# 

# In[86]:


recommend_apps('ibis Paint X', num_recommendations=5)


# In[ ]:





# ### step 18 - Save model AND preprocessing objects into file
# 

# In[88]:


# Step 18 - Save model AND preprocessing objects into file (Updates)
model_data = {
    'model': best_model,
    'scaler': scaler,
    'encoder': encoder,
    'tfidf_vectorizer': tfidf_vectorizer,
    'numerical_features': numerical_features,  # List of numerical features
    'categorical_features': categorical_features,  # List of categorical features
    'apps_list': X_train['App'].tolist()  # List of app names
}

with open('app_recommender_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Model and components saved successfully")
