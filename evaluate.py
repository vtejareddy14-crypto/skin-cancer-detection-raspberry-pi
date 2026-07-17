import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset path
root = 'd:/CNN PROJECT/melanoma_cancer_dataset'

# Transform (same as test transform used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load test dataset
test_data = datasets.ImageFolder(
    root=os.path.join(root, 'test'),
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class_names = test_data.classes

# Define model architecture (same as training)
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

# Load trained model
model = SkinCancerCNN(num_classes=2).to(device)
model.load_state_dict(torch.load('d:/CNN PROJECT/skin_cancer_model.pth', map_location=device))
model.eval()

# Collect predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# Calculate Accuracy
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# Plot Confusion Matrix (Clean Clinical Style)
plt.figure(figsize=(7,6))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size":16})

plt.title(f"Confusion Matrix\nOverall Accuracy: {accuracy*100:.2f}%",
          fontsize=16,
          pad=15)

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)

plt.tight_layout()
plt.show()

# Print Classification Report
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))
