# Model Evaluation using sklearn

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Loading iris dataset

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a model

model = LogisticRegression()

# Train the model

model.fit(X_train, y_train)

# Predict the model

y_pred = model.predict(X_test) 

# Evaluate the model

print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Precision Score: ", precision_score(y_test, y_pred, average='weighted'))
print("Recall Score: ", recall_score(y_test, y_pred, average='weighted'))

# Confusion Matrix

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Classification Report

print("Classification Report: \n", classification_report(y_test, y_pred))