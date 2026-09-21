import pandas as pd  # Importing the pandas library for data manipulation and analysis
from sklearn.utils import shuffle  # Importing the shuffle function from sklearn to shuffle the data
import numpy as np  # Importing the numpy library for numerical operations

# 1. Define the 4 manual entries as a list of lists
data = [
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor'],
    [6.3, 3.3, 6.0, 2.5, 'Iris-virginica']
]
# 2. Convert the list into a pandas DataFrame
df = pd.DataFrame(data)

# Shuffling the rows of the dataset to randomize the order of the data
df = shuffle(df)

# Displaying the rows of the shuffled dataset using the head() function
print(df.head())

# Extracting the first four columns (features) as a NumPy array
X = df.iloc[:, 0:4].values

# Extracting the fifth column (target) as a NumPy array
y = df.iloc[:, 4].values

# Print the first 4 rows of the feature matrix (X)
print(X[0:4])

# Print the first 4 rows of the target array (y)
print(y[0:4])

from sklearn.model_selection import train_test_split
# 75% for train and 25% for test
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

# Encoding the labels: 1 for setosa, 0 otherwise (Changed from -1 to 0)
train_labels = np.where(train_labels == 'Iris-setosa', 1, 0)
test_labels = np.where(test_labels == 'Iris-setosa', 1, 0)

print('Train data:', train_data[0:2])
print('Train labels:', train_labels[0:2])

print('Test data:', test_data[0:2])
print('Test labels:', test_labels[0:2])

from sklearn.linear_model import SGDClassifier
# Importing SGDClassifier to replace Perceptron so we can use a sigmoid function

sigmoid_model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.1, max_iter=100)
# Initializing the model:
# - loss='log_loss': Replaces the step function with a sigmoid (logistic) function.
# - learning_rate='constant' & eta0=0.1: Keeps the manual learning rate.
# - max_iter=10: The maximum number of iterations.

sigmoid_model.fit(train_data, train_labels)
# Training the model using the training data and labels

# Using the trained model to predict labels for the test data
test_preds = sigmoid_model.predict(test_data)
# It returns an array of predicted labels (1 for 'Iris-setosa', 0 for other species) for the test_data.

# Printing the predicted labels for the test dataset
print("Test Predictions:", test_preds)

from sklearn.metrics import accuracy_score
# Importing the accuracy_score function from sklearn's metrics module.

# Calculate the accuracy of the model by comparing predicted labels with actual test labels
accuracy = accuracy_score(test_preds, test_labels)

# Print the accuracy, rounded to two decimal places, and multiplied by 100 to show it as a percentage.
print('Accuracy:', round(accuracy, 2) * 100, "%")