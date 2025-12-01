
#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Agglomerative Hierarchical Clustering
# ================================================================


# This script demonstrates two powerful hierarchical clustering techniques:
#   1. Agglomerative Hierarchical Clustering (AHC)
#   2. BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
#
# Purpose:
# - To showcase unsupervised learning methods that group countries based on
#   geographic coordinates (Longitude, Latitude).
# - To illustrate how dendrograms and clustering algorithms reveal hidden
#   structures in data.
#
# Why It Matters (Recruiter-Friendly Narrative):
# - Clustering is widely used in market segmentation, social network analysis,
#   and anomaly detection.
# - Demonstrating mastery of clustering techniques highlights strong data science
#   fundamentals, ability to handle unsupervised learning, and skill in
#   communicating technical insights clearly.
#
# Techniques Highlighted:
# - Data loading and preprocessing with Pandas
# - Visualization with Matplotlib and Seaborn
# - Hierarchical clustering using SciPy and scikit-learn
# - Dendrogram construction for interpretability
# - Birch algorithm for scalable clustering
#
# Note on Statistical Tests:
# While this script focuses on clustering, common regression diagnostics
# (Durbin-Watson, QQ plot, Goldfeld-Quandt, VIF) are explained in comments
# to demonstrate broader statistical literacy:
#   - Durbin-Watson: Detects autocorrelation in residuals
#   - QQ Plot: Tests normality of residuals
#   - Goldfeld-Quandt: Tests heteroscedasticity
#   - VIF: Detects multicollinearity
# These are not executed here but included for recruiter context.
# ================================================================


# ==== Step 1: Import Required Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn styling improves readability of plots for professional presentation
sns.set_style('whitegrid')


# ==== Step 2: Load the Dataset ====

# Recruiter-Friendly Note:
# Pandas is the industry-standard library for data manipulation.
# Here, we load a CSV file containing country coordinates.
dataset = pd.read_csv('countrycluster.csv')

dataset  # Displaying the dataset helps confirm successful load


# ==== Step 3: Visualize Raw Data Distribution ====

# Visualization is critical for exploratory data analysis (EDA).
# Scatter plots reveal spatial distribution of countries by longitude/latitude.
plt.scatter(dataset['Longitude'], dataset['Latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()


# ==== Step 4: Transform Dataset into Feature Matrix ====

# Selecting only Longitude and Latitude columns for clustering.
# iloc is used for integer-based indexing in Pandas.
X = dataset.iloc[:, [1, 2]].values

X  # Display transformed feature matrix


# ==== Step 5: Construct Dendrogram ====

# Dendrograms are tree-like diagrams that visualize hierarchical clustering.
# They show the order and distance at which clusters are merged.
from scipy.cluster.hierarchy import dendrogram, linkage

# Linkage function computes hierarchical clustering.
# Default method is 'ward', but other linkage criteria (single, complete, average)
# can be specified depending on clustering needs.
L = linkage(X)

# Define custom labels for clarity in visualization
labels = ['USA', 'CAN', 'France', 'UK', 'GER', 'AUS']

# Plot dendrogram with labels
dendrogram(L, labels=labels)
plt.show()


# ==== Step 6: Apply Agglomerative Hierarchical Clustering ====

from sklearn.cluster import AgglomerativeClustering

# AgglomerativeClustering builds clusters bottom-up by merging closest pairs.
# distance_threshold defines the cutoff for merging clusters.
# Setting n_clusters=None allows the algorithm to determine clusters dynamically.
# #####  Agglomerative Hierarchical Clustering (AHC) is a widely used method in cluster analysis that organizes data points into a tree-like structure called a dendrogram. This method is particularly effective for understanding the relationships among data points based on their similarities.

# ##### How Agglomerative Hierarchical Clustering Works ?

#  Initialization: Each data point is initially treated as a separate cluster.
#  Distance Calculation: The dissimilarity (or distance) between each pair of clusters is computed, often using metrics like Euclidean distance.
#  Merging Clusters:
#  * The two clusters that are closest to each other are merged to form a new cluster.
#  * This process is repeated iteratively, recalculating distances as clusters are merged, until all points are grouped into a single cluster.

#  Linkage Criteria: Different methods can be used to determine how distances are calculated when merging clusters.
# 
#  Dendrogram Construction: The results of the clustering process are represented in a dendrogram, which visually depicts the order and distance at which clusters were merged.


model = AgglomerativeClustering(distance_threshold=200, n_clusters=None)

# Setting n_clusters=None means that you are not specifying a fixed number of clusters for the algorithm to create.
# Instead, the algorithm will determine the number of clusters  based on the provided distance_threshold

# Fit the model and predict cluster assignments
y_pred = model.fit_predict(X)

y_pred  # Display cluster labels


# ==== Step 7: Visualize Agglomerative Clustering Results ====

# Scatter plot with color-coded clusters.
# cmap='rainbow' ensures visually distinct cluster colors.
plt.scatter(dataset['Longitude'], dataset['Latitude'], c=y_pred, cmap='rainbow')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()


# ==== Step 8: Apply Birch Algorithm ====

# Birch is designed for large datasets and memory efficiency.
# It incrementally clusters data points, making it suitable for streaming data.

# BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is an unsupervised data mining algorithm designed for hierarchical clustering, particularly effective for large datasets.
#  Purpose: BIRCH is specifically tailored to handle large datasets that may not fit entirely in memory. It incrementally clusters incoming multi-dimensional data points while optimizing resource usage (memory and time).

# Single Scan Requirement: BIRCH typically requires only one scan of the database, making it efficient in terms of I/O operations.


from sklearn.cluster import Birch

# Setting n_clusters=3 specifies the desired number of clusters.
Model1 = Birch(n_clusters=3)

# Fit and predict cluster assignments
y_pred1 = Model1.fit_predict(X)

y_pred1  # Display Birch cluster labels


# ==== Step 9: Visualize Birch Clustering Results ====
plt.scatter(dataset['Longitude'], dataset['Latitude'], c=y_pred1, cmap='rainbow')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title('Birch Algo')
plt.show()


# ================================================================
# FINAL SUMMARY
# ================================================================
# Key Takeaways:
# - Agglomerative Hierarchical Clustering builds clusters iteratively and is
#   highly interpretable via dendrograms.
# - Birch algorithm is optimized for large datasets and streaming scenarios.
# - Both methods demonstrate unsupervised learning techniques that reveal
#   hidden structures in data without labeled outcomes.
#
# Broader Statistical Literacy (Recruiter Context):
# - Durbin-Watson: Would test for autocorrelation if this were regression.
# - QQ Plot: Would check residual normality.
# - Goldfeld-Quandt: Would test heteroscedasticity.
# - VIF: Would detect multicollinearity.
#
# Professional Impact:
# - This project highlights ability to apply clustering algorithms,
#   interpret results, and communicate insights clearly.
# - Strong visualization and narrative make this script recruiter-friendly
#   and portfolio-ready for GitHub or LinkedIn.
# ================================================================