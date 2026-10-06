#Muhammad Anees
#537630

import tensorflow as tf #building and training the CNN
from tensorflow.keras import layers, models # For creating layers and models
import matplotlib.pyplot as plt # for plotting images and accuracy/loss graphs
import numpy as np # for numerical operations


# Loading the CIFAR-10 dataset from Tensorflow datasets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values from range [0, 256] to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Plot the first 5 images from the training set along with their labels
plt.figure(figsize=(10, 20))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# Building the CNN model
model = models.Sequential([
    # FIrst convolutional layer with 32 filters, kernel size of 3x3, ReLU activation, and input shape of 32x32x3 (CIFAR-10 images)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), # First convolutional layer with 32 filters]
    # Max pooling layer with 2x2 pool
    layers.MaxPooling2D((2, 2)),
    # Second Convolutional layer with 64 filters and Relu activation
    layers.Conv2D(32, (3, 3), activation='relu'),
    # Max pooling layer with 2x2 pool
    layers.MaxPooling2D((2, 2)),
    # Third Convolutional layer with 64 filters and Relu activation
    layers.Conv2D(64, (3, 3), activation='relu'),


    # Flatten the output to feed into the fully connected layers
    layers.Flatten(),
    # Fully connected layer with 64 units and ReLU activation
    layers.Dense(32, activation='relu'),
    # Output layer with 10 units (one for each class) and softmax activation
    layers.Dense(10, activation='softmax')
])


# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Display the model architecture summary
model.summary()


# Train the model for 10 epochs and validate on the test data
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))



# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels,)
print(f"Test accuracy: {test_acc}")


# Plotting the training and validation accuracy and loss over epochs
plt.figure(figsize=(12, 4))
# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


#Make predictions on the first 5 test images
predictions = model.predict(test_images[:5])


#Define a function to display images and predictions
def plot_image(i, predictions_array, true_label, img, class_names):
    prediction_array, true_label, img = predictions_array[i], true_label[i][0], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    
    #Get the predicted label
    predicted_label = np.argmax(prediction_array)

    # Set the color of the text based on whether the prediction is correct
    color = 'blue' if predicted_label == true_label else 'red'

    #show the predicted label and true label
    plt.xlabel(f"Predicted: {class_names[predicted_label]} (True: {class_names[true_label]})", color=color)

#Visualise the first 5 test images with predictions
plt.figure(figsize=(5, 10))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plot_image(i, predictions, test_labels, test_images, class_names)

plt.tight_layout()
plt.show()


