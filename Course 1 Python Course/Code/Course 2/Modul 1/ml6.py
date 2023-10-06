# ML alogrithm on wine recognition dataset and adding model evaluation and validation part

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset from sklearn

from sklearn.datasets import load_wine
wine = load_wine()

print(wine.DESCR)

# Creating dataframe from the dataset

df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Adding target variable to the dataframe

df['target'] = wine.target

# Splitting the dataset into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), df.target, test_size=0.2)

# Training the model using Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Predicting the target variable for test set

y_pred = model.predict(X_test)

# Calculating the accuracy of the model

from sklearn.metrics import accuracy_score

print('Accuracy of the model is: ', accuracy_score(y_test, y_pred))

# Calculating the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix is: \n', cm)

# Plotting the confusion matrix

import seaborn as sns

plt.figure(figsize=(10,7))

sns.heatmap(cm, annot=True)

plt.xlabel('Predicted')

plt.ylabel('Truth')

plt.show()

# Plotting the decision tree

from sklearn import tree

plt.figure(figsize=(15,10))

tree.plot_tree(model, filled=True)

plt.show()


