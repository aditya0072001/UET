# Storytelling with Data Visualization

# Import libraries

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

# storytelling iris dataset

