#Muhammad Anees
#537630

#importing Required Libraries
import numpy as np #for Numerical Operations
import matplotlib.pyplot as plt #for Plotting
from tensorflow.keras.datasets import mnist #for Loading the MNIST Dataset
from tensorflow.keras.models import Sequential #for Creating the Model
from tensorflow.keras.layers import Dense, Flatten #for Adding Layers to the Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical #for One-Hot Encoding the Labels
from tensorflow.keras.optimizers import Adam, SGD, RMSprop #for Optimizer



# step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data() #Load the dataset
# Normalize the pixel values to be between [0 and 1]
x_train = x_train.astype('float32') / 255.0 #Normalize the training data
x_test = x_test.astype('float32') / 255.0 #Normalize the testing data
# One-hot encode the labels (0-9) for categorical classification
y_train = to_categorical(y_train, 10) #One-hot encode the training labels
y_test = to_categorical(y_test, 10) #One-hot encode the testing labels


# Step 2: Build the Neural Network Model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a 784-dimensional vector
    Dense(128, activation='relu'),   # Hidden layer with 128 neurons and ReLU activation
    Dropout(0.4),
    Dense(64, activation='relu'),    # Hidden layer with 64 neurons and ReLU activation
    Dropout(0.2),
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit) and softmax activation
])
# Print the model summary to see the architecture
model.summary() #Print the model summary


# Step 3: Compile the Model
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])


# Step 4: Train the Model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2) #Train the model


# Step 5: Evaluate the Model on Test Data
test_loss, test_accuracy = model.evaluate(x_test, y_test) #Evaluate the model on test data
print(f'Test Accuracy: {test_accuracy:.4f}') #Print the test accuracy


# Step 6: Visulize the Training and Validation Accuracy and Loss
plt.figure(figsize=(12, 5)) #Set the figure size
# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout() #Adjust the layout to prevent overlap
plt.show()


# Step 7: Make Predictions on Test Data
predictions = model.predict(x_test) #Make predictions on the test data
# Display the first test image and its predicted label
plt.figure(figsize=(5, 5)) #Set the figure size
plt.imshow(x_test[0], cmap='gray') #Display the first test image
plt.title(f'Predicted Label: {np.argmax(predictions[0])}') #Display the predicted label
plt.axis('off') #Turn off the axis
plt.show()
#Display a grid of image with true and predicted labels
num_images = 9 #Number of images to display
plt.figure(figsize=(10, 10)) #Set the figure size
for i in range(num_images): #Loop through the number of images
    plt.subplot(3, 3, i + 1) #Create a subplot for each image
    plt.imshow(x_test[i], cmap='gray') #Display the test image
    plt.title(f'True: {np.argmax(y_test[i])}, Pred: {np.argmax(predictions[i])}') #Display true and predicted labels
    plt.axis('off') #Turn off the axis
plt.tight_layout() #Adjust the layout to prevent overlap
plt.show()