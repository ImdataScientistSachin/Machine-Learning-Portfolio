#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# ================================================================

# This script demonstrates DBSCAN (Density-Based Spatial Clustering of Applications with Noise),
# a powerful unsupervised clustering algorithm.

##### Intro DBSCAN : Density-Based Spatial Clustering of Applications with Noise, is a popular clustering algorithm designed to identify clusters of arbitrary shapes in large spatial datasets. It operates based on the concept of density, which allows it to effectively discover clusters while also distinguishing noise.

# Purpose:
# - DBSCAN identifies clusters of arbitrary shapes in spatial datasets.
# - It distinguishes dense regions (clusters) from sparse regions (noise).

# Why It Matters (Recruiter-Friendly Narrative):
# - Unlike k-means, DBSCAN does not require specifying the number of clusters.
# - It is robust to outliers and can discover non-spherical clusters.
# - Demonstrating DBSCAN highlights ability to handle unsupervised learning,
#   interpret clustering parameters, and communicate results clearly.

# Key Concepts:
# - eps (ε): Maximum distance between two points to be considered neighbors.
# - min_samples (MinPts): Minimum number of points required to form a dense region (core point ).
# - Core Point: Has at least MinPts neighbors within eps radius.
# - Border Point: Lies within eps of a core point but has fewer neighbors.
# - Noise: Points that are neither core nor border.

##### Key Features of DBSCAN : 

#### Density-Based Clustering: DBSCAN defines clusters as dense regions of points separated by areas of lower density. This enables the algorithm to identify clusters that are not necessarily spherical or uniform in shape, making it versatile for various applications.
####  Input Parameters: The algorithm requires two main parameters . 

#### : Eps (ε): The maximum distance between two points for them to be considered part of the same neighborhood . 

# ================================================================


# ==== Step 1: Import Required Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # Professional plot styling


# ==== Step 2: Create Synthetic Dataset ====
# Using sklearn's make_blobs to generate sample data with 3 cluster centers.
from sklearn.datasets import make_blobs

centers = [[0.5, 2], [-1, -1], [1.5, -1]]  # Cluster centers
X, _ = make_blobs(n_samples=400, centers=centers, cluster_std=0.5, random_state=0)

X.shape  # Confirm dataset dimensions


# ==== Step 3: Visualize Raw Data Distribution ====
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1])
plt.title("Synthetic Dataset Distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ==== Step 4: Apply DBSCAN Algorithm ====

from sklearn.cluster import DBSCAN

# eps = 0.5 → radius of neighborhood
# min_samples = 20 → minimum points to form a cluster

model = DBSCAN(eps=0.5, min_samples=20)

# AS we decrease the radius resultant increase of the noise (N)
# Increasing or Higher the MinPts parameter  
# many points being classified as noise rather than forming part of a cluste



# Fit model and predict cluster labels
y_pred = model.fit_predict(X)

y_pred  # Cluster assignments (-1 indicates noise)


# ==== Step 5: Interpret Cluster Labels ====
# DBSCAN assigns:
# - Cluster labels (0, 1, 2, …) for dense regions
# - Noise points as -1
# Example: Class = 3 clusters (0,1,2), Noise = -1


# ==== Step 6: Visualize DBSCAN Clustering Results ====
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow')
plt.title("DBSCAN Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ================================================================
# FINAL SUMMARY
# ================================================================
# Key Takeaways:
# - DBSCAN successfully identified clusters of arbitrary shapes without predefining cluster count.
# - Noise points (-1) represent outliers or sparse regions.
# - eps and min_samples are critical parameters that control cluster density.
#
# Professional Impact:
# - Demonstrates ability to apply density-based clustering, interpret parameters,
#   and visualize results.
# - Highlights unsupervised learning expertise and ability to handle noisy datasets.
# - Portfolio-ready project for GitHub/LinkedIn showcasing clustering skills.
# ================================================================