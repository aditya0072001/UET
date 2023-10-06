# Advanced plotting techniques with matplotlib and seborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style to Seaborn

sns.set(style="darkgrid")

# Load the dataset iris sklearn

iris = sns.load_dataset("iris")

# Plot the distribution of the petal lengths

sns.distplot(iris["petal_length"], kde=False)
plt.show()

# Plot the distribution of the petal lengths again, this time using rugplot

sns.distplot(iris["petal_length"], kde=False, rug=True)
plt.show()

# Plot the distribution of the petal lengths using a histogram and 20 bins

sns.distplot(iris["petal_length"], bins=20, kde=False, rug=True)

plt.show()

# Plot the distribution of the petal lengths using a histogram and 50 bins

sns.distplot(iris["petal_length"], bins=50, kde=False, rug=True)

plt.show()

# Plot the distribution of the petal lengths using a kernel density estimate

sns.distplot(iris["petal_length"], hist=False, rug=True)

plt.show()









