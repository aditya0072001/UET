# naive bayes classifier on iris dataset

# importing libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# loading iris dataset
iris = load_iris()

# store features matrix in X
X = iris.data

# store response vector in y

y = iris.target

# splitting X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# training the model on training set

gnb = GaussianNB()

gnb.fit(X_train, y_train)

# making predictions on the testing set

y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)

print("Gaussian Naive Bayes model accuracy(in %):", accuracy_score(y_test, y_pred)*100)


