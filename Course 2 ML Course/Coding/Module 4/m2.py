# Cross validation on iris dataset

# Import libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a model
model = LogisticRegression()

# Cross validation
scores = cross_val_score(model, X, y, cv=5)

# Print the accuracy for each fold:
print(scores)

# Print the mean accuracy
print(scores.mean())

