# Muhammad Anees
# 537630

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Prepare Data Loaders with Augmentation and Normalization
# Pre-trained models on ImageNet expect 3-channel images normalized with specific mean and std

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Helper function to train and evaluate models
def train_model(model, criterion, optimizer, num_epochs=5):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
    return evaluate_model(model)

def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total * 100
    return accuracy

# Define the device to fix the NameError
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Custom CNN Model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

print("Training Custom CNN...")
custom_model = CustomCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_model.parameters(), lr=0.001)
acc_custom = train_model(custom_model, criterion, optimizer, num_epochs=5)
print(f"Custom Model Test Accuracy: {acc_custom:.2f}%")

# 4. Feature Extraction with ResNet50
print("Running ResNet50 Feature Extraction...")
resnet_fe = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for param in resnet_fe.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
num_ftrs = resnet_fe.fc.in_features
resnet_fe.fc = nn.Linear(num_ftrs, 10)

optimizer = optim.Adam(resnet_fe.fc.parameters(), lr=0.001)
acc_resnet_fe = train_model(resnet_fe, criterion, optimizer, num_epochs=5)
print(f"ResNet50 Feature Extraction Accuracy: {acc_resnet_fe:.2f}%")

# 5. Fine-Tuning with VGG16 (including optimization strategies)
print("Running VGG16 Fine-Tuning...")
vgg_ft = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# Unfreeze classifier layers and last block of features
for param in vgg_ft.parameters():
    param.requires_grad = False

# Unfreeze the last convolutional block of VGG16 (features.24 to features.30)
for param in vgg_ft.features[24:].parameters():
    param.requires_grad = True

# Modify VGG Classifier for 10 classes with dropout to prevent overfitting
num_ftrs = vgg_ft.classifier[6].in_features
vgg_ft.classifier[6] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 10)
)

# Using a small learning rate for fine-tuning to prevent destroying pre-trained weights
optimizer = optim.Adam([
    {'params': vgg_ft.features[24:].parameters(), 'lr': 1e-5},
    {'params': vgg_ft.classifier.parameters(), 'lr': 1e-4}
])

acc_vgg_ft = train_model(vgg_ft, criterion, optimizer, num_epochs=5)
print(f"VGG16 Fine-Tuning Accuracy: {acc_vgg_ft:.2f}%")

# 6. Compare All Results
print("=== Performance Comparison ===")
print(f"Custom CNN Accuracy: {acc_custom:.2f}%")
print(f"ResNet50 Feature Extraction Accuracy: {acc_resnet_fe:.2f}%")
print(f"VGG16 Fine-Tuning Accuracy: {acc_vgg_ft:.2f}%")