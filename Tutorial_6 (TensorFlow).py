# Muhammad Anees
# 537630

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg, decode_predictions as decode_vgg
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet, decode_predictions as decode_resnet
import numpy as np
import tensorflow as tf

# Function to Load and Preprocess Image
def load_and_preprocess_image(img_path, model_name):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224 pixels
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Preprocess image according to the model's requirements
    if model_name == 'vgg':
        return preprocess_vgg(img_array)
    elif model_name == 'resnet':
        return preprocess_resnet(img_array)

from tensorflow.keras.applications import VGG16, ResNet50
#Load pretrained VGG16 Model
vgg_model = VGG16(weights='imagenet')
# Load pretrained Resnet50 Model
resnet_model = ResNet50(weights='imagenet')

from tensorflow.keras.applications.vgg16 import decode_predictions as decode_vgg
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_resnet
# Function to make top 5 predictions
def make_predictions(model, img_array, model_name):
    preds = model.predict(img_array)  # Make predictions
    if model_name == 'vgg':
        return decode_vgg(preds, top=5)  # Decode predictions (VGG16 format)
    elif model_name == 'resnet':
        return decode_resnet(preds, top=5)  # Decode predictions (ResNet50 format)

# Predict and print top 5 guesses
def print_top_5predictions(model, img_path, mod_name):
    img_array = load_and_preprocess_image(img_path, model_name)
    predictions = make_predictions(model, img_array, model_name)

    # Print top 5 Predictions with Probabilities
    print(f"Top 5 predictions for {model_name.upper()} model:")
    for i, pred in enumerate(predictions[0]):
        print(f"{i+1}. {pred[1]} ({pred[2]*100:.2f}% probability)")

# Provide the direct path to your image on your computer
img_path = r"E:\Old Data Anees\Anees\MS Mechanical Engineering\DL\Tutorial\Tutorial 6\Image\Picture New.png"

# Predict and print top 5 using VGG16
vgg_processed = load_and_preprocess_image(img_path, 'vgg')
vgg_predictions = make_predictions(vgg_model, vgg_processed, 'vgg')
print("\nTop 5 predictions from VGG16:")
for i, pred in enumerate(vgg_predictions[0]):
    print(f"{i+1}. {pred[1]} ({pred[2]*100:.2f}% probability)")

# Predict and print top 5 using ResNet50
print("\nTop 5 predictions from ResNet50:")
resnet_processed = load_and_preprocess_image(img_path, 'resnet')
resnet_predictions = make_predictions(resnet_model, resnet_processed, 'resnet')
for i, pred in enumerate(resnet_predictions[0]):
    print(f"{i+1}. {pred[1]} ({pred[2]*100:.2f}% probability)")