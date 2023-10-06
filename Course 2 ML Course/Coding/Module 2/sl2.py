# Logistic Regression on sklearn dataset

# importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# loading the dataset

data = datasets.load_breast_cancer()

# creating a dataframe

df = pd.DataFrame(data.data, columns=data.feature_names)

# adding a column to the dataframe

df['class'] = data.target

# creating a dependent and independent variable

X = df.drop('class', axis=1)
Y = df['class']

# splitting the dataset into training and testing set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# creating a model

model = LogisticRegression()

# training the model

model.fit(X_train, Y_train)

# predicting the output

Y_pred = model.predict(X_test)

# calculating the accuracy

accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy: ", accuracy)

# predicting the output for a single input

input_data = (13.08, 15.71, 85.63, 520, 0.1075, 0.127, 0.04568, 0.0311, 0.1967, 0.06811, 0.1852, 0.7477, 1.383, 14.67, 0.004097, 0.01898, 0.01698, 0.00649, 0.01678, 0.002425, 14.5, 20.49, 96.09, 630.5, 0.1312, 0.2776, 0.189, 0.07283, 0.3184, 0.08183)

# changing the input data to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshaping the numpy array as we are predicting for only one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# predicting the output

print(model.predict(input_data_reshaped))

# predicting the probability of the output

print(model.predict_proba(input_data_reshaped))

# graph

plt.plot(input_data)

plt.show()























