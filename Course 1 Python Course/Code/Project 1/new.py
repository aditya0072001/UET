import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the stock symbol and timeframe
stock_symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2022-12-31'

# Retrieve the stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Clean and preprocess the data
stock_data = stock_data.dropna()  # Remove rows with missing values
stock_data['Date'] = pd.to_datetime(stock_data.index)  # Convert index to a datetime column

# Calculate financial metrics and indicators
stock_data['DailyReturn'] = stock_data['Close'].pct_change()  # Calculate daily returns
stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()  # Calculate 50-day simple moving average

# Visualize the stock price movements
plt.figure(figsize=(12, 6))
sns.lineplot(data=stock_data, x='Date', y='Close')
sns.lineplot(data=stock_data, x='Date', y='SMA50')
plt.title(f'Stock Price Movements for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(['Close', 'SMA50'])
plt.show()

# Perform statistical analysis
stock_returns = stock_data['DailyReturn'].dropna()
mean_return = stock_returns.mean()
std_dev = stock_returns.std()
correlation_matrix = stock_data[['Close', 'Volume']].corr()

# Print the results
print(f"Mean daily return: {mean_return:.4f}")
print(f"Standard deviation of daily returns: {std_dev:.4f}")
print(f"Correlation matrix:\n{correlation_matrix}")

# Perform exploratory data analysis (EDA)
sns.pairplot(data=stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])
plt.title('Pairplot of Stock Data')
plt.show()

sns.boxplot(data=stock_data[['Open', 'High', 'Low', 'Close']])
plt.title('Boxplot of Stock Prices')
plt.show()

sns.histplot(data=stock_data, x='Volume', kde=True)
plt.title('Distribution of Stock Volume')
plt.xlabel('Volume')
plt.ylabel('Count')
plt.show()

# Prepare the data for modeling
X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = stock_data['Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize the predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title(f'Actual vs Predicted Stock Prices for {stock_symbol}')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
