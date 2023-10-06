```python
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
```

The code documentation for the given code is as follows:

1. Import the required libraries:
   - `geopandas` for working with geospatial data
   - `matplotlib.pyplot` for plotting

2. Read the geospatial data:
   - The `gpd.datasets.get_path('naturalearth_lowres')` function provides the path to the natural earth low-resolution dataset, which contains geospatial information for countries and continents. This data is loaded using `gpd.read_file()` and stored in the `world` variable.

3. Perform exploratory data analysis (EDA):
   - Histogram: Plot the histogram of population estimates using the `hist()` function from `matplotlib.pyplot`. Customize the title, x-label, and y-label to provide meaningful information about the plot.
   - Scatter Plot: Plot the scatter plot of population estimates (`pop_est`) against land area (`landarea`) using the `plot.scatter()` function. Set the title, x-label, and y-label to describe the plot.
   - Boxplot: Create a boxplot of population estimates grouped by continent using the `boxplot()` function. Use the `by` parameter to specify the grouping variable (`continent`). Customize the title, x-label, and y-label accordingly.
   - Bar Chart: Plot a bar chart of the number of countries in each continent using the `value_counts()` function on the `continent` column and `plot(kind='bar')`. Set the title, x-label, and y-label to provide a clear understanding of the plot.

4. Plot the geospatial data:
   - Create a figure with subplots using `plt.subplots()`, specifying the number of rows and columns (2, 2

) and the figure size.
   - Choropleth Map: Set the title of the first subplot (`axs[0, 0]`) as "Choropleth Map". Use the `plot()` function on the `world` geospatial data, specifying the column to use for coloring (`pop_est`), the color map (`cmap='YlOrRd'`), and customizing the linewidth and edgecolor of the map.
   - Bubble Map: Set the title of the second subplot (`axs[0, 1]`) as "Bubble Map". Use the `plot()` function on the `world` geospatial data, specifying the column to use for coloring (`pop_est`), the color map (`cmap='YlOrRd'`), and customizing the linewidth and edgecolor of the map.

5. Display the plots using `plt.show()`.