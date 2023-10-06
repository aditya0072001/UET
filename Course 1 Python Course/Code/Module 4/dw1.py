# Data Wranglign and Transformation

# handling missing values

# import libraries

import pandas as pd

# read the data

df = pd.DataFrame({'A':[1,2,None,4,5]})

print(df)

# check for missing values

print(df.isnull())

# fill missing values with 0

df.fillna(0)

# fill missing values with mean

df.fillna(df.mean(), inplace=True)

print(df)

# fill missing values with median

df.fillna(df.median(), inplace=True)

print(df)

# fill missing values with mode

df.fillna(df.mode(), inplace=True)

print(df)

# handling outliers

# import libraries

import matplotlib.pyplot as plt

# read the data

df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9,10]})

print(df)

# check for outliers

df.boxplot(column='A')

plt.show()

# remove outliers

df = df[df['A']<8]

print(df)

# remove outliers using z-score

from scipy import stats
import numpy as np

df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9,10]})

print(df)

df = df[(np.abs(stats.zscore(df))<3).all(axis=1)]

print(df)

# remove outliers using IQR

Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)

df = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

print(df)

# handling duplicates

# import libraries

import pandas as pd

# read the data

df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9,10]})

print(df)

# add duplicate rows

df = df.append(df)

print(df)

# check for duplicates

print(df.duplicated())

# remove duplicates

df = df.drop_duplicates()

print(df)

