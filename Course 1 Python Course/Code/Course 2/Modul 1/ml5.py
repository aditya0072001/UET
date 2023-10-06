# Supervised learning and unsupervised learning example on same dataset

# importing libraries

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# loading dataset

dataset = datasets.load_digits()

X = dataset.data
y = dataset.target

# splitting dataset into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# preprocessing data

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# performing unsupervised learning on dataset using KMeans clustering

kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train_scaled)

cluster_labels = kmeans.predict(X_test_scaled)

# performing supervised learning on dataset using Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

# Evaluating the performance of the unsupervised learning model K means clustering

from sklearn.metrics import silhouette_score

print("Silhouette Score : ",silhouette_score(X_test_scaled, cluster_labels))

# Evaluating the performance of the supervised learning model Logistic Regression

print("Accuracy Score : ",accuracy_score(y_test, y_pred))