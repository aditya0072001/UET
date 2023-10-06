# PCA on sklearn dataset

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA

# Importing the dataset

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Applying PCA

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

# Creating new Dataframe with PCA results
pca_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
pca_df['target'] = y

# Plotting the results

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)

targets = [0, 1, 2]

colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indicesToKeep = pca_df['target'] == target
    ax.scatter(pca_df.loc[indicesToKeep, 'PC1'], pca_df.loc[indicesToKeep, 'PC2'], c = color, s = 50)

ax.legend(targets)

ax.grid()

plt.show()