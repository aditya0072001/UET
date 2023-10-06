# Example unsupervised learning algorithm: K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import linear_model

# Importing the dataset from sklearn

from sklearn.datasets import load_breast_cancer

# Load dataset

data = load_breast_cancer()

# Print the data

print(data)

# Print the features

print(data.feature_names)

# Print the targets

print(data.target_names)

# Print the shape of the data

print(data.data.shape)

# Print the first 5 rows of the data

print(data.data[0:5])

# Print the target values

print(data.target)

# Print the shape of the target

print(data.target.shape)

# Print the type of the target

print(data.target.dtype)

# Print the target names

print(data.target_names)

#EDA

# Create a dataframe

df = pd.DataFrame(data.data, columns=data.feature_names)

# Print the first 5 rows of the dataframe

print(df.head())

# Print the shape of the dataframe

print(df.shape)

# Print the last 5 rows of the dataframe

print(df.tail())

# Print the info of the dataframe

print(df.info())

# Print the description of the dataframe

print(df.describe())

# Print the distribution of the dataframe

print(df.describe().T)

# Print the correlation of the dataframe

print(df.corr())

# Print the heatmap of the dataframe

sns.heatmap(df.corr())
plt.show()

# Print the pairplot of the dataframe

sns.pairplot(df)
plt.show()

# Print the countplot of the dataframe

sns.countplot(data.target)
plt.show()

# Print the scatterplot of the dataframe

sns.scatterplot(x='mean radius', y='mean texture', data=df)
plt.show()

# k means clustering on breast cancer dataset

from sklearn.cluster import KMeans

X = data.data

k = 2

kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(X)

# Print the cluster centers

print(kmeans.cluster_centers_)

# Print the inertia

print(kmeans.inertia_)

# Print the labels

print(kmeans.labels_)

# Print the number of iterations

print(kmeans.n_iter_)

# Print the prediction

print(kmeans.predict(X))

# Print the score

print(kmeans.score(X))

for i,label in enumerate(kmeans.labels_):
    print("Point:", X[i], "Label:", label)

# Plot the clusters

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.show()




















