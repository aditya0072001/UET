# Overfitting and Underfitting on sklearn dataset diabetes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create linear regression object
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Calculate the training error and testing error of the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_error = mean_squared_error(y_train, y_train_pred)
test_error = mean_squared_error(y_test, y_test_pred)

print("Training error: ", train_error)
print("Testing error: ", test_error)

# Plot outputs

plt.scatter(X_test[:, 2], y_test, color='black')
plt.scatter(X_test[:, 2], y_test_pred, color='blue')
plt.show()

# Create a polynomial regression object
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Train the model using the training sets
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Calculate the training error and testing error of the model
y_train_pred = poly_model.predict(X_train_poly)
y_test_pred = poly_model.predict(X_test_poly)

train_error_poly = mean_squared_error(y_train, y_train_pred)
test_error_poly = mean_squared_error(y_test, y_test_pred)

print("Training error: ", train_error_poly)
print("Testing error: ", test_error_poly)


# Plot outputs

plt.scatter(X_test[:, 2], y_test, color='black')
plt.scatter(X_test[:, 2], y_test_pred, color='blue')
plt.show()