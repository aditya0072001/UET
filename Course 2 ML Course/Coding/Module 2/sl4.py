# SVM on Sklearn iris dataset

# Importing libraries

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Loading the iris dataset

iris = datasets.load_iris()
X = iris.data # Features
y = iris.target # Target Variable

# Split the data into training and testing sets

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.2 , random_state = 42)

# Create the SVM classifier
svm = SVC(kernel = 'linear')

# train SVM classifier

svm.fit(X_train,y_train)

# Make predictions on the testing set

y_pred = svm.predict(X_test)

# Check accuracy score

print("Accuracy score of SVM is : ", accuracy_score(y_test,y_pred))