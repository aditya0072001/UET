# Bagging and Boosting Example

# Bagging

# Importing libraries

from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading dataset

iris = load_iris()
X = iris.data
y = iris.target

# Splitting dataset into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Creating model

model = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=20)

# Training model

model.fit(X_train, y_train)

# Predicting model

y_pred = model.predict(X_test)

# Calculating accuracy

print("Accuracy:", accuracy_score(y_test, y_pred))

# Boosting

# Importing libraries

from sklearn.ensemble import AdaBoostClassifier

# Creating model

model = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=5, learning_rate=1)

# Training model

model.fit(X_train, y_train)

# Predicting model

y_pred = model.predict(X_test)

# Calculating accuracy

print("Accuracy:", accuracy_score(y_test, y_pred))