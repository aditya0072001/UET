# Import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Perform exploratory data analysis (EDA)
diabetes_df = pd.DataFrame(data=X, columns=diabetes.feature_names)
target_df = pd.DataFrame(data=y, columns=['target'])

# Plot histograms for each feature
sns.set(style='ticks')
sns.set_palette('husl')
for feature in diabetes.feature_names:
    sns.histplot(data=diabetes_df, x=feature, kde=True)
    sns.despine()
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# Plot the correlation heatmap
correlation_matrix = diabetes_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Box plots for each feature
sns.set(style='whitegrid')
sns.set_palette('pastel')
for feature in diabetes.feature_names:
    sns.boxplot(data=diabetes_df, y=feature)
    sns.despine()
    plt.title(f'{feature} Distribution')
    plt.ylabel(feature)
    plt.show()

# Pairplot for feature comparison
sns.pairplot(data=diabetes_df)
plt.title('Feature Pairplot')
plt.show()

# Countplot for target variable
sns.set(style='ticks')
sns.set_palette('husl')
sns.countplot(data=target_df, x='target')
plt.title('Diabetes Target Count')
plt.xlabel('Diabetes Progression')
plt.ylabel('Count')
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


# Deploy the model for real-time predictions
def predict_diabetes(data):
    # Preprocess the input data
    scaled_data = scaler.transform(data)

    # Make predictions using the trained model
    predictions = model.predict(scaled_data)

    return predictions

# Example usage

new_data = [[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]]
predicted_values = predict_diabetes(new_data)
print("Predicted Diabetes Progression:", predicted_values)


# Save the model

import joblib

joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'diabetes_scaler.pkl')

# Load the model

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('diabetes_scaler.pkl')








