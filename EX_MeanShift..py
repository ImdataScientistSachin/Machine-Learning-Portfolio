
#!/usr/bin/env python
# coding: utf-8

# ==================================:== ============================
# 📊 Executive Summary : Mean Shift Clustering , a partition-based unsupervised learning algorithm
# ================================================================

# This script demonstrates **Mean Shift Clustering**, a partition-based unsupervised learning algorithm.
# Unlike K-Means, Mean Shift does not require pre-specifying the number of clusters. Instead, it discovers
# clusters by locating the **modes (peaks)** of a density function.

#  Mean Shift clustering is an iterative method that aims to discover clusters by locating the modes (peaks) of a density function. It operates by shifting data points towards regions of maximum density, effectively grouping similar data points together.
#  This approach is especially useful in scenarios where the shape and size of clusters are not uniform or known in advance

#### How Does Mean Shift Work?
#   Initialization: The algorithm starts with a set of initial points (centroids), which can be selected randomly or from the dataset itself.
#  Density Estimation: For each point, it computes the density of surrounding points using a kernel function within a defined bandwidth (search window).
#  Mean Shift Vector Calculation: The algorithm calculates the mean shift vector, which points towards the direction of increasing density.
#  Point Update: Each point is moved along this vector to a new position that is closer to regions of higher density.
#  Iteration: Steps 2 to 4 are repeated until convergence, meaning that points stabilize around local maxima in the density function.



# Why Mean Shift matters:
# - Automatically determines the number of clusters.
# - Handles clusters of varying shapes and sizes.
# - Useful in applications like image segmentation, object tracking, and geographic grouping.
#
# Recruiter-Friendly Takeaway:
# This script shows practical application of **density-based clustering** with clear visualizations,
# highlighting awareness of unsupervised learning techniques and professional presentation.
# ================================================================


# ==== Step 1: Import Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift


# ==== Step 2: Load Dataset ====
dataset = pd.read_csv('countrycluster.csv')
dataset.head()


# ==== Step 3: Extract Features ====
X_org = dataset.iloc[:, 1:3].values  # Longitude and Latitude


# ==== Step 4: Generate Synthetic Data (for demonstration) ====
# make_blobs creates synthetic clusters similar to the dataset format.
n_samples = len(X_org)   # Number of samples based on dataset size
centers = 3              # Number of cluster centers
cluster_std = 3.5        # Standard deviation of clusters

X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
print("Synthetic Data Shape:", X.shape)


# ==== Step 5: Visualize Raw Data ====
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title('Synthetic Data Distribution')
plt.show()


# ==== Step 6: Apply Mean Shift Clustering ====
model = MeanShift()

try:
    y_pred = model.fit_predict(X)
    print("Cluster Predictions:", y_pred)
except Exception as e:
    print(f"An error occurred: {e}")


# ==== Step 7: Visualize Clusters ====
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow', marker='o')
plt.title('Mean Shift Clustering Results')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# ================================================================
# 📊 Conceptual Notes
# ================================================================
# - Initialization: Start with candidate centroids.
# - Density Estimation: Compute density using kernel function within bandwidth.
# - Mean Shift Vector: Points towards higher density regions.
# - Update: Move points along vector until convergence.
# - Result: Points stabilize around density peaks, forming clusters.
#
# Advantages:
# - No need to predefine number of clusters.
# - Handles irregular cluster shapes.
#
# Limitations:
# - Computationally expensive for large datasets.
# - Bandwidth parameter strongly influences results.
#
# Awareness of these tradeoffs demonstrates professional-level understanding of clustering techniques.


# ================================================================
# 📌 Final Summary
# ================================================================
# - Mean Shift clustering was applied to synthetic geographic data.
# - Algorithm discovered clusters by shifting points towards density peaks.
# - Visualization confirmed separation of clusters without predefining k.

# ================================================================