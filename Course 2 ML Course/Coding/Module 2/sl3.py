# Decision Tree and random forest algorithms on iris dataset sklearn 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the iris dataset

iris = datasets.load_iris()

# Create a dataframe with the four feature variables

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# View the top 5 rows

print(df.head())

# Add a new column with the species names, this is what we are going to try to predict

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the top 5 rows

print(df.head())

# Create a dataframe with all training data except the target column

X = df.drop(columns=['species'])

# Create a dataframe with only the target column

y = df['species']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create a decision tree and fit it to the training data

tree = DecisionTreeClassifier(random_state=1)

tree.fit(X_train, y_train)

# Check the accuracy of the model

print(tree.score(X_test, y_test))

# Create a random forest and fit it to the training data

rf = RandomForestClassifier(n_estimators=100, random_state=1)

rf.fit(X_train, y_train)

# Check its accuracy

print(rf.score(X_test, y_test))

# Get the feature importance array

importances = rf.feature_importances_

# List of tuples with variable and importance

feature_list = list(zip(df.columns, importances))

# Sort the feature importances by most important first

feature_list = sorted(feature_list, key = lambda x: x[1], reverse=True)

# Print out the feature and importances

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_list];

# New random forest with only the two most important variables

rf_most_important = RandomForestClassifier(n_estimators=100, random_state=1)

# Extract the two most important features

important_indices = [df.columns.get_loc('petal length (cm)'), df.columns.get_loc('petal width (cm)')]

train_important = X_train.iloc[:, important_indices]

test_important = X_test.iloc[:, important_indices]

# Train the random forest

rf_most_important.fit(train_important, y_train)

# Check the accuracy of the model

print(rf_most_important.score(test_important, y_test))

# Set the style

plt.style.use('fivethirtyeight')

# list of x locations for plotting

x_values = list(range(len(importances)))

# Make a bar chart

plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values, [pair[0] for pair in feature_list], rotation='vertical')

# Axis labels and title

plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')

# Show the graph

#plt.show()

# plotting random forest and decision tree

from sklearn.tree import export_graphviz
import pydotplus

dot_data = export_graphviz(tree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')

dot_data = export_graphviz(rf.estimators_[0], out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('rf0.png')

















