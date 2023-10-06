# Stock Market analysis using python

# importing the necessary libraries

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Define the stock symbol and timeframe

stock_symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2022-12-31'

# Retrieve the stock data

stock_data = yf.download(stock_symbol, start = start_date, end = end_date)

# Print the first five rows of the stock data

print(stock_data.head())

# Clean and preprocess the data

stock_data = stock_data.dropna() # Drop missing values
stock_data['Date'] = pd.to_datetime(stock_data.index) # Convert to datetime

# Calculate financial metrics and indicators

stock_data['DailyReturn'] = stock_data['Close'].pct_change() # Daily simple rate of return
stock_data['SMA50'] = stock_data['Close'].rolling(window = 50).mean() # 50-day simple moving average

print(stock_data.head())

stock_data = stock_data.dropna()

# Visualize the stock price movements

#plt.figure(figsize = (12, 6))
#sns.lineplot(data = stock_data, x='Date', y='Close')
#sns.lineplot(data = stock_data, x='Date', y='SMA50')
#plt.title('Stock Price Movements for ' + stock_symbol)
#plt.xlabel('Date')
#plt.ylabel('Price ($)')
#plt.legend(labels = ['Close', 'SMA50'])
#plt.show()

# Perform statistical analysis

stock_returns = stock_data['DailyReturn'].dropna()
mean_return = stock_returns.mean()
std_dev = stock_returns.std()
correlation_matrix = stock_data[['Close','Volume']].corr()

# Printing the results

print('The mean daily simple rate of return is {:.4f}'.format(mean_return))
print('The daily standard deviation is {:.4f}'.format(std_dev))
print('The correlation between closing price and volume is {:.4f}'.format(correlation_matrix.iloc[0,1]))

# perform EDA

#sns.pairplot(data = stock_data[['Open','High','Low','Close','Volume']])
#plt.title('Pairplot of Stock Data')
#plt.show()

#sns.boxplot(data = stock_data[['Open','High','Low','Close']])
#plt.title('Boxplot of Stock Prices')
#plt.show()

#sns.histplot(data = stock_data , x = 'Volume',kde = True)
#plt.title("Distribution of Stock's Volume")
#plt.xlabel('Volume')
#plt.ylabel('Count')
#plt.show()

sns.heatmap(data = correlation_matrix, annot = True)
plt.title('Correlation Matrix')
plt.show()

# Prepre the data for modeling
X = stock_data[['Open','High','Low','Close','Volume']].values
y = stock_data['Close'].values


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2, random_state= 42)

# Train the model
model = LinearRegression()
model.fit(X_train,y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: {:.4f}'.format(mse))

# Visualize the predicted vs actual prices
plt.figure(figsize = (12,6))
plt.plot(y_test, color = 'blue', label = 'Actual Price')
plt.plot(y_pred, color = 'red', label = 'Predicted Price')
plt.title('Predicted vs Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()





