Code Documentation for "Predicting House Prices"

1. Importing the Required Libraries:
   - pandas: Used for data manipulation and analysis.
   - numpy: Used for numerical computations.
   - seaborn: Used for data visualization.
   - matplotlib.pyplot: Used for creating plots and charts.
   - sklearn.datasets.load_boston: Used to load the Boston housing dataset.
   - sklearn.model_selection.train_test_split: Used to split the data into training and testing sets.
   - sklearn.preprocessing.StandardScaler: Used to scale the features of the dataset.
   - sklearn.linear_model.LinearRegression: Used for linear regression modeling.
   - sklearn.metrics.mean_squared_error: Used to evaluate the model performance.

2. Loading and Preparing the Data:
   - The Boston housing dataset is loaded using the load_boston function.
   - The features (X) and target (y) are assigned from the dataset.

3. Data Manipulation and EDA:
   - The data is converted to a pandas DataFrame for easier manipulation.
   - The boston_df DataFrame is created with feature names as column headers.
   - The target_df DataFrame is created with the target variable 'MEDV'.
   - Exploratory Data Analysis (EDA) is performed:
     - Histograms are plotted for each feature using seaborn's histplot function.
     - Correlation heatmap is plotted using seaborn's heatmap function.
     - Box plots are plotted for each feature using seaborn's boxplot function.
     - Pairplot is created for feature comparison using seaborn's pairplot function.

4. Data Splitting and Preprocessing:
   - The dataset is split into training and testing sets using train_test_split function.
   - The features are scaled using StandardScaler:
     - The training set features (X_train) are fitted and transformed.
     - The testing set features (X_test) are transformed using the fitted scaler.

5. Model Training and Prediction:
   - LinearRegression model is initialized.
   - The model is trained using the scaled training data (X_train_scaled) and target variable (y_train).
   - Predictions are made on the scaled testing data (X_test_scaled) using the trained model.

6. Model Evaluation:
   - Mean squared error (MSE) is calculated using mean_squared_error function, comparing the actual target values (y_test) with the predicted values (y_pred).
   - The MSE is printed as "Mean Squared Error".

