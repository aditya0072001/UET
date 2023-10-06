# Linear Regression using sklearn dataset

# Importing necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics

# Loading the dataset

#boston = datasets.load_boston(return_X_y=False)

# Splitting the dataset into train and test sets

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

#X = boston.data
#y = boston.target

# Splitting the dataset into 70% train and 30% test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1)

# Creating linear regression object

reg = linear_model.LinearRegression()

# Training the model using the training sets

reg.fit(X_train, y_train)

# Making predictions using the testing set

y_pred = reg.predict(X_test)

# Comparing actual response values (y_test) with predicted response values (y_pred)

print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))

# Plotting the regression line

plt.scatter(X_test[:, 5], y_test, color='black')

plt.plot(X_test[:, 5], y_pred, color='blue', linewidth=3)

plt.xlabel("Average number of rooms per dwelling")

plt.ylabel("House Price")

plt.show()

# Plotting the residual errors

plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color='green', s=10, label='Train data')


plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color='blue', s=10, label='Test data')

plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

plt.legend(loc='upper right')

plt.title("Residual errors")

plt.show()