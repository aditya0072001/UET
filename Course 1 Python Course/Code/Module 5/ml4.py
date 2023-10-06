# introduction to scikitlearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# load the iris dataset

iris = datasets.load_iris()

# print the iris data

print(iris.data)

# print the iris labels

print(iris.target)

# print the iris labels

print(iris.target_names)

# print the iris feature names

print(iris.feature_names)

# print the iris data shape

print(iris.data.shape)

# print the iris target shape

print(iris.target.shape)

# print the iris data type

print(iris.data.dtype)

# print the iris target type

print(iris.target.dtype)

# print the iris data type

print(type(iris.data))

# print the iris target type

print(type(iris.target))

# print the iris data type

print(type(iris))

# print the iris data type

print(iris.data)

# various sklearn ml models on iris datset

from sklearn import svm

# create a svm classifier

clf = svm.LinearSVC()

# train the classifier

clf.fit(iris.data, iris.target)

# print the classifier predictions

print(clf.predict([[5.0, 3.6, 1.3, 0.25]]))

# print the classifier predictions

print(clf.predict([[5.0, 3.6, 1.3, 0.25], [5.0, 3.6, 1.3, 0.25]]))



