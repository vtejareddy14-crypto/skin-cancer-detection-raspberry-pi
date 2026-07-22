import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2#     👈 NEW (for camera)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
class_names = ['benign', 'malignant']
  
# Model
class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Load model
model = SkinCancerCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("skin_cancer_model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 📷 STEP 1: Capture image from Pi Camera
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.start()
time.sleep(2)  # warm up
picam2.capture_file("captured.jpg")
picam2.stop()
print("Image captured!")

# 📷 STEP 2: Load captured image
image = Image.open("captured.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

label = class_names[predicted.item()]
confidence_percent = confidence.item() * 100

# Result text
if label == "benign":
    diagnosis_text = "BENIGN"
    risk_text = "Low Risk"
else:
    diagnosis_text = "MALIGNANT"
    risk_text = "Please consult a dermatologist immediately"

# Display
fig = plt.figure(figsize=(8, 9), facecolor="white")
plt.axis("off")

plt.text(0.5, 0.95, "Skin Lesion Analysis", fontsize=20, ha='center', weight='bold')
plt.text(0.5, 0.91, "Diagnostic Result", fontsize=13, ha='center', color="gray")

plt.text(0.5, 0.85, f"Diagnosis: {diagnosis_text}", fontsize=18, ha='center')
plt.text(0.5, 0.81, f"Risk Level: {risk_text}", fontsize=14, ha='center')
plt.text(0.5, 0.77, f"Model Confidence: {confidence_percent:.2f}%", fontsize=14, ha='center')

ax = fig.add_axes([0.15, 0.15, 0.7, 0.55])
ax.imshow(image)
ax.axis('off')

plt.show()
