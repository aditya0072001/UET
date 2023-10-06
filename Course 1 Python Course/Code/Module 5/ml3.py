# Model Training and Evaluation sklearn seaborn

# Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Importing the dataset

from sklearn.datasets import load_iris

iris = load_iris()

# Printing the description of the dataset

print(iris.DESCR)

# Printing the features of the dataset

print(iris.feature_names)

# Printing the target values of the dataset

print(iris.target_names)

# Printing the first five rows of the dataset

print(iris.data[0:5])

# Printing the target values of the dataset

print(iris.target)

# Printing the shape of the dataset

print(iris.data.shape)

# Printing the shape of the target values

print(iris.target.shape)

# Importing the KMeans clustering

from sklearn.cluster import KMeans

# Creating an instance of KMeans

kmeans = KMeans(n_clusters=3)

# Fitting the model with the dataset

kmeans.fit(iris.data)

# Predicting the labels of the dataset

y_kmeans = kmeans.predict(iris.data)

# Printing the predicted labels

print(y_kmeans)

# Printing the centroids of the clusters

print(kmeans.cluster_centers_)

# Plotting the scatter plot of the dataset

plt.scatter(iris.data[:,0], iris.data[:,1], c=y_kmeans, cmap='rainbow')

# Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')

# Labelling the axes

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

# Plotting the title

plt.title('Sepal Length vs Sepal Width')

# Displaying the plot

plt.show()






