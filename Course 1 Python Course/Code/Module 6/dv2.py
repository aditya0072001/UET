# Geospatial Data Visualization using Geopandas on famous dataset

# Importing libraries

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

# Setting working directory

#os.chdir("C:/Users/HP/Desktop/Python/Geospatial Data Visualization")

# Reading the dataset from url for geopandas


url = "https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv"

# Reading the dataset

df = pd.read_csv(url)

# Checking the head of the dataset

print(df.head())

# Checking the tail of the dataset

print(df.tail())

# Checking the shape of the dataset

print(df.shape)

# Checking the columns of the dataset

print(df.columns)

# Checking the datatypes of the dataset

print(df.dtypes)

# Checking the info of the dataset

print(df.info())

# Checking the summary of the dataset

print(df.describe())

# Checking the missing values in the dataset

print(df.isnull().sum())

# graph ploting using geopandas

import geopandas as gpd
import matplotlib.pyplot as plt

# read the geospatial data using geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# plot the geospatial data

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# plot the world map

world.plot(ax=axs[0, 0])
#plt.show()


# Plot 1 : Choropleth Map

axs[0,0].set_title("Choropleth Map")
world.plot(column='pop_est',cmap='YlOrRd',ax=axs[0,0],linewidth = 0.8, edgecolor ='0.8',legend=True)
#plt.show()

# Plot 2 : Bubble Map

axs[0,1].set_title("Bubble Map")
world.plot(column='pop_est',cmap='YlOrRd',ax=axs[0,1],linewidth = 0.8, edgecolor ='0.8',legend=True)
plt.show()







