# K Means Clustering on SKlearn dataset

# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Loading the iris dataset

iris = datasets.load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns = iris.feature_names)
df['target'] = y

# Visualising the data

plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c = df['target'])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Original Data Points')
plt.show()

# Applying KMeans to the dataset
k = 3
kmeans = KMeans(n_clusters = k)
kmeans.fit(X)

# Add cluster labels to the dataframe

df['cluster'] = kmeans.labels_

# Visualising the clusters

plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c = df['cluster'])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('K-Means Clustering')
plt.show()

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

# Visualising the clusters

plt.scatter(X_pca[:, 0], X_pca[:, 1], c = df['cluster'])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('K-Means Clustering')
plt.show()