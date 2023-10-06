# handling inconsistent data

# import libraries

import pandas as pd

# read the data

df = pd.DataFrame({'A':['a','b','c','d','e','f','g','h','i','j']})

print(df)

# replace values

df['A'] = df['A'].replace('a','A')

print(df)

# rename columns

df = df.rename(columns={'A':'B'})

print(df)

# handling data types

# import libraries

import pandas as pd

# read the data

df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9,10]})

print(df)


# change data type

df['A'] = df['A'].astype('float')

print(df)

# data transformation technqiues

# import libraries

import pandas as pd

# read the data

df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9,10]})

print(df)

# apply function

df['B'] = df['A'].apply(lambda x: x*2)

print(df)

# map function

df['C'] = df['A'].map(lambda x: x*2)

print(df)

# applymap function

df = df.applymap(lambda x: x*2)

print(df)

# replace function

df['A'] = df['A'].replace(10,100)

print(df)

# drop function

df = df.drop(['A'], axis=1)

print(df)

# dropna function

df = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9,10],

                     'B':[11,12,13,14,15,16,17,18,19,20],})

print(df)

df = df.dropna()

print(df)

# fillna function

df = df.fillna(0)

print(df)

# fillna function

df = df.fillna(method='ffill')

print(df)

# fillna function

df = df.fillna(method='bfill')

print(df)

# fillna function

df = df.fillna(method='pad')

print(df)

# fillna function

df = df.fillna(method='backfill')

print(df)

# fillna function

df = df.fillna(method='ffill', limit=1)

print(df)

# fillna function

df = df.fillna(method='bfill', limit=1)

print(df)

# fillna function

df = df.fillna(method='pad', limit=1)

print(df)

# fillna function

df = df.fillna(method='backfill', limit=1)

print(df)

# fillna function

df = df.fillna(method='ffill', limit=2)

print(df)

# fillna function

df = df.fillna(method='bfill', limit=2)

print(df)

# fillna function

df = df.fillna(method='pad', limit=2)


print(df)

# fillna function

df = df.fillna(method='backfill', limit=2)

print(df)


# merging joining and reshaping datasets

print("merging joining and reshaping datasets")

# import libraries

import pandas as pd

# read the data

df1 = pd.DataFrame({'A':[1,2,3,4,5], 'B':[11,12,13,14,15]})

df2 = pd.DataFrame({'A':[6,7,8,9,10], 'B':[16,17,18,19,20]})

print(df1)

print(df2)

# merge function
print("merge function on a column")

df = pd.merge(df1, df2, on='A')

print(df)

# merge function

print("merge function on b column")

df = pd.merge(df1, df2, on='B')

print(df)

# merge function

df = pd.merge(df1, df2, on='B', how='left')

print(df)

# merge function

df = pd.merge(df1, df2, on='B', how='right')

print(df)

# merge function

df = pd.merge(df1, df2, on='B', how='outer')

print(df)

# merge function

df = pd.merge(df1, df2, on='B', how='inner')

print(df)

# join function

df = df1.join(df2)

print(df)

# join function

df = df1.join(df2, how='left')

print(df)

# join function

df = df1.join(df2, how='right')

print(df)

# join function

df = df1.join(df2, how='outer')

print(df)

# join function

df = df1.join(df2, how='inner')

print(df)

# concat function

df = pd.concat([df1, df2])

print(df)

# concat function

df = pd.concat([df1, df2], axis=1)

print(df)

# concat function

df = pd.concat([df1, df2], axis=1, join='inner')

print(df)

# concat function

df = pd.concat([df1, df2], axis=1, join='outer')

print(df)

# concat function

df = pd.concat([df1, df2], axis=1, join='outer', ignore_index=True)

print(df)

# concat function

df = pd.concat([df1, df2], axis=1, join='outer', ignore_index=False)

print(df)

# concat function

df = pd.concat([df1, df2], axis=1, join='outer', ignore_index=True)

print(df)

# concat function

df = pd.concat([df1, df2], axis=1, join='outer', ignore_index=False)

print(df)

# concat function

df = pd.concat([df1, df2], axis=1, join='outer', ignore_index=True)





