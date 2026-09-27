#Muhammad Anees
#537630
#Step1: Importing Required Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt #Import for plotting the learning Curves

#Step2: LOading and Splitting the Data Set
iris = load_iris() 
X = iris.data # Features
y = iris.target # labels

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Step3: Data Scaling
Scaler = StandardScaler() # Create a StandardScaler object
X_train_scaled = Scaler.fit_transform(X_train) # Fit and transform the training data
X_test_scaled = Scaler.transform(X_test) # Transform the testing data using the same scaler

#Step4: Creating and Training the MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42, learning_rate_init=0.001) # Create an MLPClassifier with two hidden layers of 10 neurons each
mlp.fit(X_train_scaled, y_train) # Train the model on the scaled training data

#Step5: Making Predictions and evaluating the model
y_pred = mlp.predict(X_test_scaled) # Make predictions on the scaled test data
accuracy = accuracy_score(y_test, y_pred) # Calculate the accuracy of the model
print(f"Accuracy of the MLP Classifier: {accuracy:.2f}") # Print accuracy of MLP Classifier
print("Classification Report:\n", classification_report(y_test, y_pred)) # Print the classification report

#Step6: Displaying the MLP Structure and training Information
print("\nMLP Structure:")
print(f"Number of layers: {mlp.n_layers_}") # Print the number of layers
print(f"Number of outputs: {mlp.n_outputs_}") # Print the number of outputs
print(f"Activation function: {mlp.activation}") # Print the activation function used
print(f"Output activation function: {mlp.out_activation_}") # Print the output activation function used
print(f"Number of epochs: {mlp.n_iter_}") # Print the number of epochs the model was trained for

#Step7: Visulizing the Learning Curves
plt.figure(figsize=(8, 6)) # Create a new figure for the plot
plt.plot(mlp.loss_curve_, label='Training Loss') # Plot the training loss curve
plt.title('MLP Classifier Learning Curve') # Set the title of the plot
plt.xlabel('Epochs') # Set the x-axis label
plt.ylabel('Loss') # Set the y-axis label
plt.legend() # Add a legend to the plot
plt.grid() # Add a grid to the plot
plt.show() # Display the plot
