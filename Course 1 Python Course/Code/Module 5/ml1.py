# Introduction to Machine Learning using Python

# Importing the libraries

import numpy as np

# Importing the dataset iris

from sklearn.datasets import load_iris

iris = load_iris() # Loading the dataset

# Printing the description of the dataset

#print(iris.DESCR)

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

# Importing the Logistic Regression classifier

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression() # Instantiating the classifier

logreg.fit(iris.data, iris.target) # Fitting the classifier

# Predicting the target values

print(logreg.predict([[5.1, 3.5, 1.4, 0.2]]))

# Predicting the target values

print(logreg.predict([[5.9, 3.0, 5.1, 1.8]]))

# Importing the train_test_split function

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=4)

# Printing the shape of the training data

print(X_train.shape)

# Printing the shape of the testing data

print(X_test.shape)

# Printing the shape of the training data





