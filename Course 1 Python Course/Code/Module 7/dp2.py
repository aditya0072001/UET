# building end-to-end data workflow using sklearn and on iris dataset

# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset

from sklearn.datasets import load_iris

iris = load_iris()

# create dataframe

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# add target column

df['target'] = iris.target

# add target names

df['target_names'] = df.target.apply(lambda x: iris.target_names[x])

# check for null values

print(df.isnull().sum())

# check for duplicate values

print(df.duplicated().sum())

# check for outliers

df.boxplot()

plt.show()

# check for correlation

df.corr()

plt.show()

# check for skewness

df.skew()

plt.show()

# check for distribution

df.hist()

plt.show()

# check for data types

print(df.dtypes)

# check for unique values

print(df.nunique())

# check for value counts

print(df.target_names.value_counts())

# check for summary statistics

print(df.describe())

# check for data imbalance

print(df.target_names.value_counts(normalize=True))

# check for data imbalance using bar plot

df.target_names.value_counts().plot(kind='bar')

plt.show()

# check for data imbalance using pie chart

df.target_names.value_counts().plot(kind='pie')

plt.show()

# check for data imbalance using count plot

import seaborn as sns

sns.countplot(df.target_names)

plt.show()

# check for data imbalance using count plot

sns.countplot(df.target_names, hue=df.target_names)

plt.show()

