#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : clustering techniques
# ================================================================

# This script demonstrates clustering techniques using the Iris dataset:
# - Dimensionality reduction with PCA
# - Clustering with KMeans
# - Optimal cluster selection using the Elbow Method
# - Density-based clustering with MeanShift

# Purpose:
# - To explore unsupervised learning methods on a classic dataset.
# - To visualize clusters in reduced dimensions for interpretability.

# Why It Matters (Recruiter-Friendly Narrative):
# - PCA + clustering is widely used in exploratory data analysis, customer segmentation,
#   and pattern discovery.
# - Demonstrating multiple clustering approaches highlights versatility in unsupervised learning.

# Key Concepts:
# - PCA: Projects high-dimensional data into fewer dimensions while preserving variance.
# - KMeans: Groups data into k clusters by minimizing within-cluster variance.
# - Elbow Method: Helps determine optimal number of clusters.
# - MeanShift: Density-based clustering that automatically finds cluster centers.

# ================================================================


# ==== Step 1: Import Required Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
sns.set_style('whitegrid')
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


# ==== Step 2: Load Iris Dataset ====

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data   # Features: Sepal length, Sepal width, Petal length, Petal width


# ==== Step 3: Apply PCA (3 Components) ====

# Reduce dimensionality from 4 features → 3 components for 3D visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

X_pca.shape  # Confirm reduced shape


# ==== Step 4: Visualize PCA Distribution ====

plt.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA Reduced Data Distribution')
plt.show()


# ==== Step 5: Apply KMeans Clustering ====
from sklearn.cluster import KMeans

# KMeans with 4 clusters (unsupervised, not tied to true Iris species count)
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)


# ==== Step 6: Visualize KMeans Clusters in 3D ====

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter points for each cluster
for i in range(4):
    ax.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], X_pca[y_kmeans == i, 2])

# Plot centroids in PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], c='black')

# Labels and Title
ax.set_title('3D Clustering of Iris Species with PCA Reduced Features')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.legend()
plt.show()


# ==== Step 7: Elbow Method for Optimal Clusters ====

WCSS = []  # Within-Cluster Sum of Squares
for i in range(1, 12):
    KM = KMeans(n_clusters=i)
    KM.fit_predict(X)
    WCSS.append(KM.inertia_)

plt.plot(range(1, 12), WCSS)
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# ==== Step 8: Apply MeanShift Clustering ====
from sklearn.cluster import MeanShift

model = MeanShift()
y_pred = model.fit_predict(X)

y_pred  # Display cluster labels


# ==== Step 9: Visualize MeanShift Clusters ====

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 3], cmap='rainbow')
# Alternative: ax.scatter(X[:,0], X[:,1], X[:,2], c=y_pred, cmap='rainbow')
plt.title("MeanShift Clustering Visualization")
plt.show()


# ================================================================
# FINAL SUMMARY
# ================================================================

# Key Takeaways:
# - PCA reduced Iris dataset to 3D for visualization.
# - KMeans grouped data into 4 clusters; centroids plotted in PCA space.
# - Elbow Method suggested optimal cluster count by analyzing WCSS.
# - MeanShift provided density-based clustering without predefining cluster count.
#
# Professional Impact:
# - Demonstrates ability to combine dimensionality reduction, clustering, and evaluation.
# - Highlights versatility in unsupervised learning approaches.
#
# ================================================================