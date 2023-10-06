# handling datetime data in python pandas

import datetime as dt
import pandas as pd

# create a datetime object

mydate = dt.datetime(2018, 1, 1, 12, 30, 59)

print(mydate)

# create a datetime object from a string

mydate = dt.datetime.strptime('2018-01-01 12:30:59', '%Y-%m-%d %H:%M:%S')

print(mydate)

# using pandas to create a datetime object

mydate = pd.to_datetime('2018-01-01 12:30:59')

print(mydate)

# using pandas to create a datetime object from a string

mydate = pd.to_datetime('2018-01-01 12:30:59', format='%Y-%m-%d %H:%M:%S')

print(mydate)



