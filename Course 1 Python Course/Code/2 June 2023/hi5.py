import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# EDA on iris dataset

iris = sns.load_dataset('iris')

print(iris.head())

print(iris.info())

print(iris.describe())

print(iris['species'].value_counts())

#sns.pairplot(iris, hue='species', palette='Dark2')

#plt.show()

setosa = iris[iris['species'] == 'setosa']

#sns.kdeplot(setosa['sepal_width'], cmap='plasma', shade=True, shade_lowest=False)

#plt.show()

# Data Cleaning and Preprocessing

#sns.heatmap(iris.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#plt.show()

#sns.boxplot(x='species', y='petal_length', data=iris)

#plt.show()

# Matplotlib and Seaborn for Data Visualization

#sns.violinplot(x='species', y='petal_length', data=iris, palette='rainbow')

#plt.show()

#sns.scatterplot(x='petal_length', y='petal_width', data=iris, hue='species')
#plt.xlabel('Petal Length')
#plt.ylabel('Petal Width')
#plt.title('Petal Length vs Width')
#plt.show()

#sns.boxplot(x='species', y='petal_length', data=iris, palette='rainbow')
#plt.xlabel('Species')
#plt.ylabel('Petal Length')
#plt.title('Petal Length vs Species')
#plt.show()

#sns.heatmap(iris.corr(), annot=True, cmap='coolwarm')

#plt.show()

#sns.histplot(iris['sepal_width'], bins=30, kde=True)

#plt.show()

#sns.barplot(x='species', y='petal_length', data=iris, palette='rainbow')

#plt.show()

#plt.pie(iris['species'].value_counts(), labels=iris['species'].unique(), autopct='%1.1f%%', shadow=True, explode=[0.1, 0.1, 0.1])

#plt.show()

#sns.lineplot(x='petal_length', y='petal_width', data=iris, hue='species')

#plt.show()

#sns.histplot(iris['petal_length'], bins=30, kde=True)

#plt.show()

# Statistical Analysis using Python

mean_sep_len = np.mean(iris['sepal_length'])
print(mean_sep_len)

corr = iris['sepal_length'].corr(iris['sepal_width'])
print(corr)

# using scipy

from scipy.stats import pearsonr

corr, p = pearsonr(iris['sepal_length'], iris['sepal_width'])

print(corr, p)

# using pandas

corr = iris['sepal_length'].corr(iris['sepal_width'])

print(corr)

# using numpy

corr = np.corrcoef(iris['sepal_length'], iris['sepal_width'])

print(corr)

# hypothesis testing

from scipy.stats import ttest_ind

setosa = iris[iris['species'] == 'setosa']
virginica = iris[iris['species'] == 'virginica']

t, p = ttest_ind(setosa['sepal_length'], virginica['sepal_length'])

print(t, p)

# regression analysis

from sklearn.linear_model import LinearRegression

X = iris[['sepal_length']]
y = iris[['sepal_width']]

model = LinearRegression()

model.fit(X, y)

print(model.coef_, model.intercept_)

plt.scatter(X, y)

plt.plot(X, model.predict(X), color='red')

plt.show()

# correlation analysis

from scipy.stats import pearsonr

corr, p = pearsonr(iris['sepal_length'], iris['sepal_width'])

print(corr, p)

# classification analysis

from sklearn.model_selection import train_test_split

X = iris.drop('species', axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

# clustering analysis

from sklearn.cluster import KMeans

X = iris.drop('species', axis=1)

model = KMeans(n_clusters=3)

model.fit(X)

print(model.labels_)

print(model.cluster_centers_)

sns.scatterplot(x='petal_length', y='petal_width', data=iris, hue=model.labels_)

plt.show()

# dimensionality reduction

from sklearn.decomposition import PCA

X = iris.drop('species', axis=1)

model = PCA(n_components=2)

model.fit(X)

print(model.explained_variance_ratio_)

print(model.components_)

X_new = model.transform(X)

print(X_new)

sns.scatterplot(x=X_new[:, 0], y=X_new[:, 1], hue=iris['species'])

plt.show()














