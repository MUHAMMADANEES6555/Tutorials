#Muhammad Anees
#537630
import numpy as np

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        """Calculate the net input: dot product of features and weights plus bias"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after applying the sigmoid function and then a threshold (1 if sigmoid >= 0.5, else -1)"""
        # Old code (step function):
        # return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

        # Calculate the net input
        net_input = self.weighted_sum(X)

        # Apply the sigmoid activation function
        sigmoid_output = 1.0 / (1.0 + np.exp(-net_input))

        # Return class label after applying a threshold to the sigmoid output
        # return np.where(sigmoid_output >= 0.5, 1, -1)
        return np.where(sigmoid_output >= 0.5, 1, 0)

    def fit(self, X, y):
        """Fit training data and update weights based on the perceptron learning rule"""
        self.w_ = np.zeros(1 + X.shape[1])  # Weights initialized to zeros (+1 for bias)
        self.errors_ = []  # List to track the number of errors in each iteration

        print("Weights:", self.w_)

        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                update = self.eta * (target - y_pred)

                # Update weights and bias
                self.w_[1:] = self.w_[1:] + update * xi
                self.w_[0] = self.w_[0] + update

                error += int(update != 0.0)

            self.errors_.append(error)
        return self

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.metrics import accuracy_score

# Step 5: Loading and preparing the Iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)

# Display the first few rows
print("Dataset Head:")
print(df.head())

# Extracting features (X) and target labels (y)
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# Splitting the data into train (75%) and test (25%) sets
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

# Encoding the labels: 1 for 'Iris-setosa', -1 otherwise
train_labels = np.where(train_labels == 'Iris-setosa', 1, -1)
test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

print('\nTrain data:', train_data[0:2])
print('Train labels:', train_labels[0:2])
print('Test data:', test_data[0:2])
print('Test labels:', test_labels[0:2])

# Step 6: Training the Perceptron model
perceptron = SklearnPerceptron(eta0=0.1, max_iter=10)
perceptron.fit(train_data, train_labels)

# Step 7: Making predictions on test data
test_preds = perceptron.predict(test_data)
print('\nPredicted test labels:', test_preds)

# Step 8: Measuring Performance
accuracy = accuracy_score(test_preds, test_labels)
print('Accuracy:', round(accuracy, 2) * 100, "%")

# Task: Take manual input for the four features
print("--- Manual Iris Flower Prediction ---")
try:
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))

    # Wrap the input into a 2D array as required by scikit-learn predict()
    manual_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict using the trained model
    prediction = perceptron.predict(manual_input)[0]

    # Display the result
    if prediction == 1:
        print("\nPrediction: It is Iris-setosa (Label: 1)")
    else:
        print("\nPrediction: It is NOT Iris-setosa (Label: 0)")

except ValueError:
    print("Invalid input! Please enter numeric values only.")