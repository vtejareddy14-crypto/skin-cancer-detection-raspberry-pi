import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
watch_folder = "d:/CNN PROJECT/input_images"
model_path = "d:/CNN PROJECT/skin_cancer_model.pth"

class_names = ['benign', 'malignant']

# Model Definition (same as training)
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

# Load Model
model = SkinCancerCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print(" Monitoring folder for new images...\n")

processed_files = set()

while True:
    files = os.listdir(watch_folder)

    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(watch_folder, file)

            if full_path not in processed_files:

                print(f"\n Image detected: {file}")
                print("Analyzing...")

                image = Image.open(full_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                label = class_names[predicted.item()]
                confidence_percent = confidence.item() * 100

                print("\n==============================")
                print("SKIN LESION ANALYSIS RESULT")
                print("==============================")

                if label == "benign":
                    print("Diagnosis: BENIGN")
                else:
                    print("Diagnosis: MALIGNANT")

                print(f"Model Confidence: {confidence_percent:.2f}%")
                print("==============================\n")

                # Show UI
                plt.imshow(image)
                plt.title(f"Diagnosis: {label.upper()}\nConfidence: {confidence_percent:.2f}%")
                plt.axis("off")
                plt.show()

                processed_files.add(full_path)

    time.sleep(2)
