# SVM with poly kernel on iris dataset

# Importing libraries

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Spliting the data into training and testing split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# Create SVM kernel model

svm_model = SVC(kernel = 'poly', degree = 8, gamma = 'auto', C = 1.0)

# Fit the model

svm_model.fit(X_train, y_train)

# Predict the model

y_pred = svm_model.predict(X_test)

# Accuracy of the model

print("Accuracy of the model is: ", svm_model.score(X_test, y_test))