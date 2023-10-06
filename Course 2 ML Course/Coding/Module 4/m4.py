# GridSearchCV and Random Search CV on sklearn iris dataset

# Importing the libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the random forest classifier

rfc = RandomForestClassifier()

# Define the grid search parameters

grid_params = {
    'n_estimators': [10, 100, 200, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

# Create the grid search

gs = GridSearchCV(estimator=rfc, param_grid=grid_params, cv=5, n_jobs=-1, verbose=1)

# Fit the grid search

gs.fit(X_train, y_train)

# Print the best parameters

print(gs.best_params_)


# Create the random search

rs = RandomizedSearchCV(estimator=rfc, param_distributions=grid_params, cv=5, n_jobs=-1, verbose=1)

# Fit the random search

rs.fit(X_train, y_train)

# Print the best parameters

print(rs.best_params_)

# Print the best score

print(rs.best_score_)

