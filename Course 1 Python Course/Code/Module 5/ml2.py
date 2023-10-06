# supervised and unsupervised learning on iris dataset using scikit-learn and seaborn

# Importing the libraries

import numpy as np

# Importing the dataset iris

from sklearn.datasets import load_iris

iris = load_iris() # Loading the dataset

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

# Importing the KNN classifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1) # Instantiating the classifier

knn.fit(iris.data, iris.target) # Fitting the classifier

# Predicting the target values

print(knn.predict([[5.1, 3.5, 1.4, 0.2]]))

# Predicting the target values

print(knn.predict([[5.9, 3.0, 5.1, 1.8]]))

# graph for knn classifier

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()

# Loading the iris dataset

iris = sns.load_dataset("iris")

# Plotting the iris dataset

sns.pairplot(iris, hue="species", height=2.5)

plt.show()

# kmeans clustering

# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

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

kmeans = KMeans(n_clusters=3) # Instantiating the classifier

kmeans.fit(iris.data) # Fitting the classifier

# Printing the labels

print(kmeans.labels_)

# Printing the predicted values

print(kmeans.predict([[5.1, 3.5, 1.4, 0.2]]))

# Printing the predicted values

print(kmeans.predict([[5.9, 3.0, 5.1, 1.8]]))

# graph for kmeans clustering

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()

# Loading the iris dataset

iris = sns.load_dataset("iris")

# Plotting the iris dataset

sns.pairplot(iris, hue="species", height=2.5)

plt.show()