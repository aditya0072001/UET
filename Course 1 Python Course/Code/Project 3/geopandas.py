# Importing the required libraries
import geopandas as gpd
import matplotlib.pyplot as plt

# Read the geospatial data using geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Perform exploratory data analysis (EDA)
# Plot the histogram of population estimates
world['pop_est'].hist(bins=20)
plt.title('Population Estimates Histogram')
plt.xlabel('Population Estimates')
plt.ylabel('Frequency')
plt.show()

# Plot the scatter plot of population estimates and land area
world.plot.scatter(x='pop_est', y='landarea')
plt.title('Population Estimates vs Land Area')
plt.xlabel('Population Estimates')
plt.ylabel('Land Area')
plt.show()

# Plot the boxplot of population estimates by continent
world.boxplot(column='pop_est', by='continent', vert=False)
plt.title('Population Estimates by Continent')
plt.xlabel('Population Estimates')
plt.ylabel('Continent')
plt.show()

# Plot the bar chart of the number of countries by continent
world['continent'].value_counts().plot(kind='bar')
plt.title('Number of Countries by Continent')
plt.xlabel('Continent')
plt.ylabel('Number of Countries')
plt.show()

# Plot the geospatial data
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Choropleth Map
axs[0, 0].set_title("Choropleth Map")
world.plot(column='pop_est', cmap='YlOrRd', ax=axs[0, 0], linewidth=0.8, edgecolor='0.8', legend=True)

# Plot 2: Bubble Map
axs[0, 1].set_title("Bubble Map")
world.plot(column='pop_est', cmap='YlOrRd', ax=axs[0, 1], linewidth=0.8, edgecolor='0.8', legend=True)

# Display the plots
plt.show()
