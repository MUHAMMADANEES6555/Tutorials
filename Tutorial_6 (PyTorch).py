# Muhammad Anees
# 537630

import torch
import torchvision.models as models
from torchvision.models import VGG16_Weights, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# Provide the direct path to your image on your computer
img_path = r"E:\Old Data Anees\Anees\MS Mechanical Engineering\DL\Tutorial\Tutorial 6\Image\Picture New.png"

# Function to Load and Preprocess Image for PyTorch
def load_and_preprocess_image(img_path):
    # Open image using PIL and ensure it's RGB
    img = Image.open(img_path).convert('RGB')

    # Standard ImageNet transformation pipeline for PyTorch models
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    # Apply transforms and add batch dimension [1, 3, 224, 224]
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

# Load pretrained models in evaluation mode
print("\nLoading pretrained VGG16 and ResNet50 models (PyTorch)...")
vgg_model = models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg_model.eval()  # Set model to evaluation mode

resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model.eval()  # Set model to evaluation mode

# Preprocess the uploaded image
img_tensor = load_and_preprocess_image(img_path)

# Function to make top 5 predictions
def make_predictions(model, tensor_input, weights_enum):
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(tensor_input)
        # Apply Softmax to get probabilities
        probabilities = F.softmax(outputs[0], dim=0)
        # Get top 5 probabilities and their class indices
        top5_prob, top5_indices = torch.topk(probabilities, 5)

        # Extract class names directly from the torchvision weights metadata
        categories = weights_enum.meta["categories"]

        predictions = []
        for i in range(5):
            class_name = categories[top5_indices[i].item()]
            prob = top5_prob[i].item()
            predictions.append((class_name, prob))

        return predictions

# Predict and print top 5 using VGG16
print("\nTop 5 predictions from VGG16:")
vgg_predictions = make_predictions(vgg_model, img_tensor, VGG16_Weights.DEFAULT)
for i, (class_name, prob) in enumerate(vgg_predictions):
    print(f"{i+1}. {class_name} ({prob*100:.2f}% probability)")

# Predict and print top 5 using ResNet50
print("\nTop 5 predictions from ResNet50:")
resnet_predictions = make_predictions(resnet_model, img_tensor, ResNet50_Weights.DEFAULT)
for i, (class_name, prob) in enumerate(resnet_predictions):
    print(f"{i+1}. {class_name} ({prob*100:.2f}% probability)")