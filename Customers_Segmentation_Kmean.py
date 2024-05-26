



import os
os.environ['OMP_NUM_THREADS'] = '1'

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read DataSet from csv file
data = pd.read_csv('B:\MY Documents\Customers_Segmentation_Kmean.csv')

# Select features for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data for better clustering results
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the number of clusters (use the Elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method results
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid()
plt.show()

# Based on the Elbow method, select the optimal number of clusters
optimal_k = 5

# Perform K-means clustering with the selected number of clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(10, 8))
for cluster in range(optimal_k):
    plt.scatter(X_scaled[data['Cluster'] == cluster][:, 0],
                X_scaled[data['Cluster'] == cluster][:, 1],
                label=f'Cluster {cluster + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.grid()
plt.show()
