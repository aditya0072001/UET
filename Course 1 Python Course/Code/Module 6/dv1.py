# interative data visualization using plotly and boken on famous dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from chart_studio import plotly
import plotly.offline as pyo

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, LogColorMapper

from sklearn import datasets

# load iris dataset
iris = datasets.load_iris()

# convert to pandas dataframe

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target']) 

# convert target to string

df['target'] = df['target'].map({0.0:'setosa', 1.0:'versicolor', 2.0:'virginica'})

# plot using matplotlib

plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
#plt.show()

# plot using plotly

scatter = go.Scatter(x=df['sepal length (cm)'], y=df['sepal width (cm)'])

layout = go.Layout(title='Iris Dataset', xaxis=dict(title='sepal length (cm)'), yaxis=dict(title='sepal width (cm)'))

fig = go.Figure(data=[scatter], layout=layout)

iplot(fig)

# Render the plot
plot_div = pyo.plot(fig, include_plotlyjs=False, output_type='div')

# Display the plot in the console
#print(plot_div)

# plot using bokeh



output_file("iris.html")

source = ColumnDataSource(data=dict(
    x=df['sepal length (cm)'],
    y=df['sepal width (cm)'],
    target=df['target']
))

plot = figure(title='Iris Sepal Width vs. Sepal Length', x_axis_label='sepal length (cm)',
              y_axis_label='sepal width (cm)')

plot.circle('x', 'y', color='target', legend_field='target', source=source)

hover = HoverTool(tooltips=[('Target', '@target')])

plot.add_tools(hover)

show(plot)


































