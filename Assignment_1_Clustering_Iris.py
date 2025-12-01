

#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : (PCA) and clustering (KMeans) using the classic Iris dataset.
# ================================================================

# This script demonstrates dimensionality reduction (PCA) and clustering (KMeans)
# using the classic Iris dataset.
#
# Purpose:
# - Reduce high-dimensional data into fewer components for visualization.
# - Apply KMeans clustering to group Iris species based on features.
#
# Why It Matters (Recruiter-Friendly Narrative):
# - PCA is a powerful technique for feature extraction and visualization.
# - KMeans is a widely used clustering algorithm for unsupervised learning.
# - Combining PCA with clustering demonstrates ability to handle dimensionality
#   reduction, clustering, and visualization in one workflow.
#
# Key Concepts:
# - PCA: Projects data into lower dimensions while preserving variance.
# - VarianceThreshold: Removes low-variance features to improve clustering.
# - KMeans: Groups data into k clusters by minimizing within-cluster variance.
# ================================================================


# ==== Step 1: Import Required Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # Professional plot styling
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


# ==== Step 2: Load Iris Dataset ====
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data   # Features: Sepal length, Sepal width, Petal length, Petal width


# ==== Step 3: Apply PCA (3 Components) ====
from sklearn.decomposition import PCA

# Reduce dimensionality from 4 features → 3 components for 3D visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)


# ==== Step 4: Apply KMeans Clustering ====
from sklearn.cluster import KMeans

# KMeans with 4 clusters (note: Iris has 3 species, but clustering is unsupervised)
kmeans = KMeans(n_clusters=4, random_state=0)
y_pred = kmeans.fit_predict(X_pca)

y_pred  # Display cluster labels


# ==== Step 5: Visualize Clusters in 3D (PCA Reduced) ====
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter points for each cluster
for i in range(4):
    ax.scatter(X_pca[y_pred == i, 0], X_pca[y_pred == i, 1], X_pca[y_pred == i, 2], label=f'Cluster {i}')

# Plot centroids in PCA space
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_pca)
centroids_pca = kmeans.cluster_centers_
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], c='black', s=50, marker='*')

# Labels and Title
ax.set_title('3D Clustering of Iris Species with 3 Features (PCA Reduced)')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.show()


# ==== Step 6: Feature Selection with VarianceThreshold ====
from sklearn.feature_selection import VarianceThreshold

# Remove low-variance features (threshold=0.5)
var_thresh = VarianceThreshold(threshold=0.5)
X_var_filtered = var_thresh.fit_transform(X)


# ==== Step 7: Apply PCA (2 Components) ====
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_var_filtered)

# Apply KMeans clustering again
kmeans = KMeans(n_clusters=4, random_state=0)
y_pred = kmeans.fit_predict(X_pca)

y_pred  # Display cluster labels


# ==== Step 8: Visualize Clusters in 2D (PCA Reduced) ====
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter points for each cluster with colors
ax.scatter(X_pca[y_pred == 0, 0], X_pca[y_pred == 0, 1], c='purple', label='Iris-setosa')
ax.scatter(X_pca[y_pred == 1, 0], X_pca[y_pred == 1, 1], c='orange', label='Iris-versicolor')
ax.scatter(X_pca[y_pred == 2, 0], X_pca[y_pred == 2, 1], c='green', label='Iris-virginica')

# Plot centroids
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_pca)
centroids_pca = kmeans.cluster_centers_
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', s=100, marker='*')

# Labels and Title
ax.set_title('3D Clustering of Iris Species with 2 Features (PCA Reduced)')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')

# Adjust view angle for better visibility
ax.view_init(elev=20, azim=30)
plt.show()

# Save plot for portfolio use
plt.savefig('distribution_plot.png')
plt.show()


# ================================================================
# FINAL SUMMARY
# ================================================================

# Key Takeaways:
# - PCA reduces dimensionality, enabling visualization of high-dimensional data.
# - VarianceThreshold improves clustering by removing low-information features.
# - KMeans groups Iris species into clusters, though unsupervised clustering
#   may not perfectly align with true species labels.
#
# Professional Impact:
# - Demonstrates ability to combine feature selection, dimensionality reduction,
#   clustering, and visualization.
# - Highlights unsupervised learning expertise and ability to communicate results.


# ================================================================