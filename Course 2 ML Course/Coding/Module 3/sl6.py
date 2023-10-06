# Hierarchical Clustering

# Importing libraries

import numpy as np
import pandas as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Importing dataset

iris = datasets.load_iris()
X = iris.data

# Perform hierarchical clustering

agg_clustering = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
agg_labels = agg_clustering.fit_predict(X)

# Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dend = dendrogram(linkage(X,method = 'ward'))
plt.axhline(y=7,color='r',linestyle='--')
#plt.show()

# Scatter plot of the clustered data points colored by cluster
plt.scatter(X[agg_labels == 0][:,2],X[agg_labels == 0][:,3])
plt.scatter(X[agg_labels == 1][:,2],X[agg_labels == 1][:,3])
plt.scatter(X[agg_labels == 2][:,2],X[agg_labels == 2][:,3])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Hierarchical Clustering')
plt.legend()
plt.show()