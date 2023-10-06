# Example Supervised Learning Alogrithm: Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# Importing the dataset from sklearn

from sklearn.datasets import load_breast_cancer

# Load dataset

data = load_breast_cancer()

# Print the data

print(data)

# Print the description of the dataset

print(data.DESCR)

# Print the features

print(data.feature_names)

# Print the labels

print(data.target_names)

# Print the shape of the data

print(data.data.shape)

# Print the labels

print(data.target.shape)

# Print the first five rows of the data

print(data.data[0:5])

# Print the first five rows of the labels

print(data.target[0:5])

# Create a dataframe

df = pd.DataFrame(np.c_[data.data, data.target], columns = np.append(data.feature_names, ['target']))

# Print the dataframe

print(df.head())

# Print the last five rows of the data

print(df.tail())

# Print the shape of the data

print(df.shape)

# Print the number of rows and columns

print(df.info())    

# Print the statistical measures of the data

print(df.describe())

# Visualize the count of number of Malignant(M) and Benign(B) cells

plt.figure(figsize=(8, 6))
plt.title('Count of Malignant(M) and Benign(B) cells')
#sns.countplot(df['target'])
#plt.show()

# Visualize the correlation

plt.figure(figsize=(20, 10))
#sns.heatmap(df.corr(), annot=True)
#plt.show()

# Create the pairplot

sns.pairplot(df, hue='target', vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
plt.show()

# logsitic regression for breast cancer

# Create the X and Y datasets for training

X = df.drop(['target'], axis=1)
Y = df['target']

# Split the data into 80% training and 20% testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Create the model

model = linear_model.LogisticRegression()

# Train the model

model.fit(X_train, Y_train)

# Print the predictions on the test data

predictions = model.predict(X_test)

print(predictions)

# Check the precision, recall, f1-score

print(classification_report(Y_test, predictions))

# Print the actual values

print(Y_test.values)

# Print the predicted values

print(model.predict(X_test))

# Check the model accuracy

print(accuracy_score(Y_test, predictions))

# Print the confusion matrix

print(confusion_matrix(Y_test, predictions))

# Print the ROC curve

y_pred_proba = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(Y_test, y_pred_proba)

auc = roc_auc_score(Y_test, y_pred_proba)

plt.plot(fpr, tpr, label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()

# Print the coefficients of the model

print(model.coef_)



















