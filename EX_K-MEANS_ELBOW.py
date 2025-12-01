#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary : K-Means Elbow Method
# ================================================================

# This script demonstrates the **K-Means Elbow Method**, a technique used to determine the optimal
# number of clusters in K-Means clustering.

# # K-MEANS_ELBOW()

#  The K-Means Elbow Method is a widely used technique in machine learning for determining the optimal number of clusters in K-means clustering.

#  The Elbow Method helps identify the appropriate number of clusters (denoted as k) by plotting the Within-Cluster Sum of Squares (WCSS) against differen values of k. 

#  The WCSS measures the variance within each cluster, calculated as the sum of squared distances between each point and its corresponding cluster centroid.d.

# ##### Why Use the Elbow Method?
#  Visual Insight: The method provides a clear visual representation, making it easier to decide on an optimal number of clusters.
# Efficiency: It helps avoid overfitting by preventing an excessive number of clusters that could lead to noise rather than meaningful patterns in data..


# Why the Elbow Method matters:
# - Provides a **visual heuristic** for selecting the right number of clusters (k).
# - Prevents overfitting by avoiding too many clusters that capture noise instead of meaningful patterns.
# - Helps balance **model simplicity** and **accuracy** in unsupervised learning tasks.

# ================================================================


# ==== Step 1: Import Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')


# ==== Step 2: Load Dataset ====
dataset = pd.read_csv('countrycluster.csv')
dataset.head()


# ==== Step 3: Visualize Raw Data ====
plt.scatter(dataset['Longitude'], dataset['Latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title('Country Distribution (Before Clustering)')
plt.show()


# ==== Step 4: Prepare Features ====
X = dataset.iloc[:, 1:3].values  # Selecting Longitude and Latitude


# ==== Step 5: Apply K-Means (Initial Example with k=3) ====
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

# Predictions
y_pred = model.fit_predict(X)

# Add cluster labels to dataset
dataset['cluster_predicted'] = y_pred
dataset.head()

# Visualize clustered distribution
plt.scatter(dataset['Longitude'], dataset['Latitude'], c=y_pred, cmap='rainbow')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title('Country Distribution (Clustered with k=3)')
plt.show()


# ==== Step 6: Elbow Method (WCSS Calculation) ====
wcss = []
for i in range(1, 7):  # Testing cluster sizes from 1 to 6
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss_itr = kmeans.inertia_  # Inertia = Within-Cluster Sum of Squares (WCSS)
    wcss.append(wcss_itr)

# WCSS values show how tightly points are grouped within clusters.
# Lower WCSS → better clustering, but too many clusters may overfit.


# ==== Step 7: Plot Elbow Curve ====
plt.plot(range(1, 7), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.show()


# ================================================================
# 📊 Conceptual Notes
# ================================================================
# - WCSS (Within-Cluster Sum of Squares): Measures variance within clusters.
# - Elbow Point: The value of k where WCSS starts to decrease more slowly,
#   forming an "elbow" in the curve.
# - Interpretation: The elbow suggests the optimal number of clusters.
#
# Limitations:
# - Elbow method is heuristic, not exact.
# - Different datasets may produce less clear elbows.
# - Should be combined with domain knowledge and other metrics (e.g., Silhouette Score).
#
# Awareness of these nuances demonstrates professional-level understanding of clustering evaluation.


# ================================================================
# 📌 Final Summary
# ================================================================
# - K-Means clustering was applied to geographic data (Longitude, Latitude).
# - The Elbow Method was used to determine optimal k by plotting WCSS vs k.
# - Visualization showed how WCSS decreases as k increases, with an elbow indicating the best choice.
#
# ================================================================