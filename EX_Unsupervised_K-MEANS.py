#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 📊 Executive Summary : K-MEANS (Partion- Based Clustering Algorithm )
# ================================================================

# K-Means clustering is a widely used unsupervised learning algorithm that groups data points into distinct clusters based on their similarities.

# This script demonstrates **K-Means Clustering**, a partition-based unsupervised learning algorithm.
# K-Means groups data points into clusters based on similarity, using iterative refinement of centroids.

##   What is K-Means Clustering?

#  Initialization: Randomly select K initial centroids from the dataset.
#  Assignment: Assign each data point to the nearest centroid based on a distance metric (commonly Euclidean distance)
#   Update Centroids: Calculate new centroids as the mean of all points assigned to each cluster.
#   Repeat: Continue assigning points and updating centroids until the centroids no longer change significantly or a maximum number of iterations is reached.t.dataset

# Why K-Means matters:
# - Helps discover hidden patterns in unlabeled data.
# - Useful in customer segmentation, geographic grouping, and market analysis.
# - Demonstrates ability to apply unsupervised learning with visualization for interpretability.

# ================================================================


# ==== Step 1: Import Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# ==== Step 2: Load Dataset ====
dataset = pd.read_csv('countrycluster.csv')
dataset.head()


# ==== Step 3: Visualize Raw Data ====
# Scatter plot of Longitude vs Latitude before clustering.
plt.scatter(dataset['Longitude'], dataset['Latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title('Country Distribution (Before Clustering)')
plt.show()


# ==== Step 4: Prepare Features ====
# Selecting Longitude and Latitude for clustering.
X = dataset.iloc[:, 1:3].values


# ==== Step 5: Apply K-Means Clustering ====
from sklearn.cluster import KMeans

# Initialize KMeans with 3 clusters.
model = KMeans(n_clusters=3, random_state=42)

# Fit model and predict cluster assignments.
y_pred = model.fit_predict(X)

# Add cluster labels to dataset.
dataset['cluster_predicted'] = y_pred
dataset.head()


# ==== Step 6: Visualize Clusters ====
plt.scatter(dataset['Longitude'], dataset['Latitude'], c=y_pred, cmap='rainbow')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title('Country Distribution (Clustered)')
plt.show()


# ================================================================
# 📊 Conceptual Notes
# ================================================================
# - Initialization: Randomly select K centroids.
# - Assignment: Assign points to nearest centroid (Euclidean distance).
# - Update: Recalculate centroids as mean of assigned points.
# - Repeat: Continue until centroids stabilize or max iterations reached.
#
# Limitations:
# - Sensitive to initial centroid placement.
# - Requires pre-specifying K (number of clusters).
# - Assumes spherical clusters of similar size.
# Awareness of these limitations demonstrates professional-level understanding of clustering techniques.


# ================================================================
# 📌 Final Summary
# ================================================================
# - K-Means clustered countries into 3 groups based on geographic coordinates.
# - Visualization showed clear separation of clusters.
# - Demonstrates ability to apply unsupervised learning for pattern discovery.

# ================================================================