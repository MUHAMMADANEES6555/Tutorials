#MUhammad Anees
#537630

#Import necessery libraries
import matplotlib.pyplot as plt #display images
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img #image preprocessing
from tensorflow.keras.utils import img_to_array #convert image to array
from tensorflow.keras.models import Sequential # Neural network model container
from tensorflow.keras.layers import Flatten, Dense # Network
import os #for creating directories and handling file system operations


# Define the folder where the images will be saved
save_folder = r"E:\Old Data Anees\Anees\MS Mechanical Engineering\DL\Tutorial\Tutorial 4\Saved_Images"

#create the folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)


# Initialize the ImageDataGenerator class with the augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40, # Randomly rotate images upto 40 degrees
    shear_range=0.2, # Randomly apply shearing (Distorting the shape)
    zoom_range=0.2, # Randomly zoom in or out on images by 20%
    horizontal_flip=True, # Randomly flip images horizontally
    brightness_range=(0.5, 1.5))# Randomly change the brightness of images


#Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)), #Flatten 28x28 images  to a ID vector of 784 features
    Dense(128, activation='relu'), #Fully connected layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'), #Fully connected layer with 64 neurons and ReLU activation
    Dense(10, activation='softmax') #Output layer with 10 neurons (for 10 classes) and softmax activation
])

#Print the model summary to see the architecture
model.summary()


# Load the sample image from the given path
img = load_img(r"E:\Old Data Anees\Anees\MS Mechanical Engineering\DL\Tutorial\Tutorial 4\Sample_Images\Picture New.png") # Load the image

# Convert the image to a numpy array
X = img_to_array(img)


#Reshape the image to add an extra dimension for batch Processing
X = X.reshape((1,) + X.shape)


#Initialize a counter to limit the number of augmented images generated
i = 0

#Loop to generate and save augmented images
for batch in datagen.flow(X, batch_size=1, 
                          save_to_dir=save_folder, 
                            save_prefix='image',
                            save_format='png'):
    # Convert the batch to an image (for visualization)
    augmented_image = batch[0].astype('uint8') # Convert the batch to an image (for visualization)

    #display the augmented image using matplotlib
    plt.figure() # Create a new figure for each augmented image
    plt.imshow(augmented_image) # Display the augmented image
    plt.axis('off') # Turn off axis labels and ticks
    plt.show() # Show the plot

    # Increment the counter
    i += 1

    # Stop after generating and saving 20 imges
    if i >= 20:
        break


    # Print the location of saved augmented images
    print(f"Augmented images saved in the folder: {os.path.abspath(save_folder)}")