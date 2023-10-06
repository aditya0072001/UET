import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Convert the data to a pandas DataFrame for easier manipulation
boston_df = pd.DataFrame(data=X, columns=boston.feature_names)
target_df = pd.DataFrame(data=y, columns=['MEDV'])

# Perform exploratory data analysis (EDA)
# Plot histograms for each feature
sns.set(style='ticks')
sns.set_palette('husl')
for feature in boston.feature_names:
    sns.histplot(data=boston_df, x=feature, kde=True)
    sns.despine()
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# Plot the correlation heatmap
correlation_matrix = boston_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Box plots for each feature
sns.set(style='whitegrid')
sns.set_palette('pastel')
for feature in boston.feature_names:
    sns.boxplot(data=boston_df, y=feature)
    sns.despine()
    plt.title(f'{feature} Distribution')
    plt.ylabel(feature)
    plt.show()

# Pairplot for feature comparison
sns.pairplot(data=boston_df)
plt.title('Feature Pairplot')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
