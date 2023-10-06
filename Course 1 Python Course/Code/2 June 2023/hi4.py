import pandas as pd

# Dataframe Basics

# Create a dataframe from a dictionary

data = {'Name': ['John', 'Tim', 'Rob'], 'Age': [34, 23, 42]}

df = pd.DataFrame(data)

print(df)

# Data Indexing and Selection

# Selecting a single column

print(df['Name'])

# Selecting multiple columns

print(df[['Name', 'Age']])

# Selecting rows by index

print(df.iloc[0])

# Selecting rows by index range

print(df.iloc[0:2])

# Data Manipulation

df.sort_values(by='Age', inplace=True)

print(df)

# Data Aggregation

print(df['Age'].mean())

# Data Visualization

# print(df.plot.bar(x='Name', y='Age'))

# Data Import and Export

# df.to_csv('data.csv')

df = pd.read_csv('data.csv')

print(df)

# Data Cleaning

df.dropna(inplace=True)

print(df)

# Data Transformation

df['Age'] = df['Age'].apply(lambda x: x + 1)

print(df)

# Time Series Analysis

df['Date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')

print(df)

# Excel files

df.to_excel('data.xlsx')

df = pd.read_excel('data.xlsx')

# json files

df.to_json('data.json')

df = pd.read_json('data.json')

# SQL databases

import sqlite3

conn = sqlite3.connect('data.db')

df.to_sql('data', conn)

df = pd.read_sql('SELECT * FROM data', conn)

print(df)







