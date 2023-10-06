# Hyperparameter tuning on sklearn dataset iris

# Importing the libraries

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Creating the classifier

classifier = RandomForestClassifier()

# Creating the parameters grid

parameters = [{'n_estimators': [10, 100, 1000], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}]

# Creating the Grid Search

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

# Fitting the model

grid_search = grid_search.fit(X_train, y_train)

# Getting the best accuracy

best_accuracy = grid_search.best_score_

# Getting the best parameters

best_parameters = grid_search.best_params_


# Predicting the Test set results

y_pred = grid_search.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Making the accuracy score

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

# Making the classification report

from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)

print("Best accuracy: ", best_accuracy , "Best Parameters: ", best_parameters, "Confusion Matrix: ", cm, "Accuracy: ", accuracy, "Classification Report: ", report)