import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def main():
    # ======================
    # DEVICE
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ======================
    # SPEED BOOST
    # ======================
    torch.backends.cudnn.benchmark = True

    # ======================
    # TRANSFORMS
    # ======================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # ======================
    # DATASET
    # ======================
    base_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(base_dir, "melanoma_cancer_dataset", "train")
    test_path  = os.path.join(base_dir, "melanoma_cancer_dataset", "test")

    train_data = datasets.ImageFolder(train_path, transform=transform)
    test_data  = datasets.ImageFolder(test_path, transform=transform)

    print("Classes:", train_data.classes)

    # ⚠️ num_workers=0 → avoids Windows crash
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # ======================
    # MODEL (MobileNetV2)
    # ======================
    model = models.mobilenet_v2(weights=None)  # no download
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    print("Model running on:", next(model.parameters()).device)

    # ======================
    # LOSS + OPTIMIZER
    # ======================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ======================
    # TRAINING
    # ======================
    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # ======================
    # TEST
    # ======================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # ======================
    # SAVE MODEL
    # ======================
    torch.save(model.state_dict(), "mobilenet_model.pth")
    print("Model saved successfully!")

# ======================
# RUN SAFE (Windows fix)
# ======================
if __name__ == "__main__":
    main()