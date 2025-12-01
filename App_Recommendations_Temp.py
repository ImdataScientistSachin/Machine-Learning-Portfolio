#!/usr/bin/env python
# coding: utf-8


# App_Recommendations_Temp ready for deployment (works fine )

# Builts as per deployment



# --- Notebook Setup: Imports and Global Settings ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle  # For saving the model and preprocessing objects

warnings.filterwarnings("ignore")



# --- 1. Data Loading and Preprocessing Functions ---

def load_and_preprocess_data(file_path):
    """Loads and preprocesses the data."""
    dataset = pd.read_csv(file_path)

    # Correctly handle the 'Free' value
    dataset = dataset[dataset['Installs'] != 'Free']

    dataset['Installs'] = dataset['Installs'].astype(str).str.replace(r'[+,]', '', regex=True).astype(int)
    dataset['Price'] = dataset['Price'].str.replace('$', '', regex=False).astype(float)
    dataset['Last Updated'] = pd.to_datetime(dataset['Last Updated'], errors='coerce')
    dataset.dropna(subset=['Last Updated'], inplace=True)
    dataset['Reviews'] = pd.to_numeric(dataset['Reviews'], errors='coerce')
    dataset['Size'] = dataset['Size'].str.replace('M', '').str.replace('k', '').str.replace('Varies with device', 'NaN').astype(float)
    dataset['App Age'] = (pd.to_datetime('2018-12-31') - dataset['Last Updated']).dt.days
    dataset['App Age'] = dataset['App Age'].fillna(dataset['App Age'].median())
    dataset['Features'] = dataset['Category'] + ' ' + dataset['Genres'] + ' ' + dataset['App']
    dataset = dataset.dropna(subset=['Rating'])
    return dataset



# --- 2. Data Splitting Function ---

def create_train_test_data(dataset):
    """Creates training and testing data."""
    X = dataset.drop('Rating', axis=1)
    y = dataset['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# --- 3. Feature Fitting and Transformation Functions ---

def fit_and_transform_features(X_train, X_test, numerical_features, categorical_features):
    """Fits and transforms the features."""
    # Imputation
    for col in numerical_features:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_test[col].median())

    # Scaling
    scaler = MinMaxScaler()
    numerical_scaled_train = scaler.fit_transform(X_train[numerical_features])
    numerical_scaled_test = scaler.transform(X_test[numerical_features])

    # Encoding
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data_train = encoder.fit_transform(X_train[categorical_features])
    encoded_data_test = encoder.transform(X_test[categorical_features])

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(X_train['Features'])
    tfidf_matrix_test = tfidf_vectorizer.transform(X_test['Features'])

    return numerical_scaled_train, numerical_scaled_test, encoded_data_train, encoded_data_test, tfidf_matrix_train, tfidf_matrix_test, scaler, encoder, tfidf_vectorizer


# --- 4. Sparse Matrix Creation Function ---
def create_sparse_matrix(numerical_scaled_train, numerical_scaled_test, encoded_data_train, encoded_data_test):
    """Creates sparse matrix."""
    numerical_scaled_train_sparse = csr_matrix(numerical_scaled_train)
    numerical_scaled_test_sparse = csr_matrix(numerical_scaled_test)
    encoded_data_train_sparse = csr_matrix(encoded_data_train)
    encoded_data_test_sparse = csr_matrix(encoded_data_test)
    return numerical_scaled_train_sparse, numerical_scaled_test_sparse, encoded_data_train_sparse, encoded_data_test_sparse
    



# --- 5. Feature Combination Function ---

def combine_features(tfidf_matrix_train, numerical_scaled_train_sparse, encoded_data_train_sparse, tfidf_matrix_test, numerical_scaled_test_sparse, encoded_data_test_sparse):
    """Combines features."""
    combined_features_train = hstack([tfidf_matrix_train, numerical_scaled_train_sparse, encoded_data_train_sparse])
    combined_features_test = hstack([tfidf_matrix_test, numerical_scaled_test_sparse, encoded_data_test_sparse])
    return combined_features_train, combined_features_test



# --- 6. Model Training Function ---

def train_model(combined_features_train, y_train):
    """Trains the model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(combined_features_train, y_train)
    return model


# --- 7. Model Evaluation Function ---

def evaluate_model(model, combined_features_test, y_test):
    """Evaluates the model."""
    y_pred = model.predict(combined_features_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
    return mse, mae, r2


# --- 8. Model Saving Function ---

def save_model_and_objects(model, scaler, encoder, tfidf_vectorizer, model_filename="random_forest_model.pkl", scaler_filename="minmax_scaler.pkl", encoder_filename="onehot_encoder.pkl", vectorizer_filename="tfidf_vectorizer.pkl"):
    """Saves the model and preprocessing objects to disk."""
    pickle.dump(model, open(model_filename, 'wb'))
    pickle.dump(scaler, open(scaler_filename, 'wb'))
    pickle.dump(encoder, open(encoder_filename, 'wb'))
    pickle.dump(tfidf_vectorizer, open(vectorizer_filename, 'wb'))
    print("Model and preprocessing objects saved successfully.")


# --- 9. Model Loading Function ---

def load_model_and_objects(model_filename="random_forest_model.pkl", scaler_filename="minmax_scaler.pkl", encoder_filename="onehot_encoder.pkl", vectorizer_filename="tfidf_vectorizer.pkl"):
    """Loads the model and preprocessing objects from disk."""
    model = pickle.load(open(model_filename, 'rb'))
    scaler = pickle.load(open(scaler_filename, 'rb'))
    encoder = pickle.load(open(encoder_filename, 'rb'))
    tfidf_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
    return model, scaler, encoder, tfidf_vectorizer



# --- 10. App Recommendation Function ---

def recommend_apps(app_name, X_train, X, combined_features_train, model, numerical_features, scaler, encoder, tfidf_vectorizer, num_recommendations=5):
    """
    Recommends similar apps based on feature similarity using the trained model.
    """
    try:
        # Get the features of the target app
        app_index = X[X['App'] == app_name].index[0]

        # Isolate the app's data
        app_data = X.loc[[app_index]].copy()

        # Impute numerical features using the training data's median
        for col in numerical_features:
            app_data[col] = app_data[col].fillna(X_train[col].median())

        # Transform the test data
        numerical_scaled_app = scaler.transform(app_data[numerical_features])
        encoded_data_app = encoder.transform(app_data[categorical_features])
        tfidf_matrix_app = tfidf_vectorizer.transform(app_data['Features'])

        # Convert to sparse matrix
        numerical_scaled_app_sparse = csr_matrix(numerical_scaled_app)
        encoded_data_app_sparse = csr_matrix(encoded_data_app)

        # Combine features
        combined_features_app = hstack([tfidf_matrix_app, numerical_scaled_app_sparse, encoded_data_app_sparse])

        # Predict rating
        predicted_rating = model.predict(combined_features_app)[0]
        print(f"Predicted rating for the app: {predicted_rating:.2f}")

        # Calculate similarity scores against all apps in X_train
        similarity_scores = cosine_similarity(combined_features_app, combined_features_train)

        # Get indices of top N similar apps
        similar_app_indices = similarity_scores.argsort()[0][-(num_recommendations+1):-1][::-1]  # Exclude the app itself

        # Get the app names
        recommended_apps = X_train.iloc[similar_app_indices]['App'].values
        print(f"Recommended apps similar to {app_name}:")
        for app in recommended_apps:
            print(app)

    except IndexError:
        print(f"App '{app_name}' not found in the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")



# --- 11. Rating Prediction Function (for Deployment) ---
def predict_rating(app_data, model, numerical_features, scaler, encoder, tfidf_vectorizer, X_train):
    """Predicts the rating of a given app using the loaded model."""
    # Convert App Data from DICT into Pandas
    app_data = pd.DataFrame([app_data])

    # Impute numerical features using the training data's median
    for col in numerical_features:
        app_data[col] = app_data[col].fillna(X_train[col].median())

    # Transform the test data
    numerical_scaled_app = scaler.transform(app_data[numerical_features])
    encoded_data_app = encoder.transform(app_data[categorical_features])
    tfidf_matrix_app = tfidf_vectorizer.transform(app_data['Features'])

    # Convert to sparse matrix
    numerical_scaled_app_sparse = csr_matrix(numerical_scaled_app)
    encoded_data_app_sparse = csr_matrix(encoded_data_app)

    # Combine features
    combined_features_app = hstack([tfidf_matrix_app, numerical_scaled_app_sparse, encoded_data_app_sparse])
    prediction = model.predict(combined_features_app)[0]
    return prediction



# --- Notebook Execution ---


# --- 1. Load and Preprocess Data ---
dataset = load_and_preprocess_data('googleplaystore.csv')
dataset.head()



# --- 2. Create Training and Testing Data ---
X_train, X_test, y_train, y_test = create_train_test_data(dataset)


# --- 3. Feature Fitting and Transformation ---

numerical_features = ['Reviews', 'Size', 'Installs', 'Price', 'App Age']
categorical_features = ['Type', 'Content Rating']
numerical_scaled_train, numerical_scaled_test, encoded_data_train, encoded_data_test, tfidf_matrix_train, tfidf_matrix_test, scaler, encoder, tfidf_vectorizer = fit_and_transform_features(X_train, X_test, numerical_features, categorical_features)


# --- 4. Create Sparse Matrices ---
numerical_scaled_train_sparse, numerical_scaled_test_sparse, encoded_data_train_sparse, encoded_data_test_sparse = create_sparse_matrix(numerical_scaled_train, numerical_scaled_test, encoded_data_train, encoded_data_test)



# --- 5. Combine Features ---
combined_features_train, combined_features_test = combine_features(tfidf_matrix_train, numerical_scaled_train_sparse, encoded_data_train_sparse, tfidf_matrix_test, numerical_scaled_test_sparse, encoded_data_test_sparse)



# --- 6. Train the Model ---
model = train_model(combined_features_train, y_train)



# --- 7. Evaluate the Model ---
mse, mae, r2 = evaluate_model(model, combined_features_test, y_test)

# --- 8. Save the Model and Objects ---
save_model_and_objects(model, scaler, encoder, tfidf_vectorizer)


# --- 9. Test the Model ---
X = dataset.drop('Rating', axis=1)
load_model_and_objects()
recommend_apps('ibis Paint X', X_train, X, combined_features_train, model, numerical_features, scaler, encoder, tfidf_vectorizer, num_recommendations=5)
