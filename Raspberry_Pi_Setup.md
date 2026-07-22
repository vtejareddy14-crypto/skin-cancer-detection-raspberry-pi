# 🍓 Raspberry Pi Deployment Guide

This guide explains the deployment process used to run the trained skin cancer detection model on a Raspberry Pi 4/5.

---

# Hardware Requirements

- Raspberry Pi 4 / Raspberry Pi 5
- Raspberry Pi OS (Bookworm/Bullseye)
- Raspberry Pi Camera Module v2
- Laptop/Desktop
- microSD Card (16 GB or higher)
- Stable Wi-Fi connection

---

# Software Requirements

Install the following on the Raspberry Pi.

- Python 3
- pip
- OpenCV
- PyTorch
- Pillow
- NumPy
- Tkinter
- VNC Server
- SSH

---

# Step 1 – Update Raspberry Pi

Open Terminal and execute:

```bash
sudo apt update
sudo apt upgrade -y
```

---

# Step 2 – Install Required Packages

```bash
sudo apt install python3-pip python3-tk python3-opencv unzip
```

Install Python libraries.

```bash
pip3 install -r requirements.txt
```

If installing manually:

```bash
pip3 install torch torchvision pillow numpy opencv-python
```

---

# Step 3 – Enable SSH

SSH is required to transfer files from the laptop.

Run:

```bash
sudo raspi-config
```

Navigate to:

```
Interface Options
    └── SSH
            └── Enable
```

---

# Step 4 – Enable VNC

To remotely access the Raspberry Pi desktop:

```bash
sudo raspi-config
```

Navigate to

```
Interface Options
    └── VNC
            └── Enable
```

Restart the Raspberry Pi if prompted.

---

# Step 5 – Find Raspberry Pi IP Address

Run:

```bash
hostname -I
```

Example:

```
192.168.1.20
```

---

# Step 6 – Connect Using VNC Viewer

Open **VNC Viewer** on your laptop.

Connect using:

```
192.168.1.20
```

Login using the Raspberry Pi username and password.

After successful login, the Raspberry Pi desktop can be controlled remotely from the laptop.

---

# Step 7 – Transfer Project Files

The project files and trained model were transferred from the laptop to the Raspberry Pi using **Secure Copy Protocol (SCP)**.

Transfer the complete project folder.

```bash
scp -r skin-cancer-detection-raspberry-pi teja-pi@192.168.1.20:/home/teja-pi/
```

Transfer an individual file.

```bash
scp filename.py teja-pi@192.168.1.20:/home/teja-pi/
```

After entering the Raspberry Pi password, the files will be copied to the Raspberry Pi.

---

# Step 8 – Navigate to the Project

```bash
cd ~/skin-cancer-detection-raspberry-pi
```

Verify the files.

```bash
ls
```

---

# Step 9 – Run the Model

Run the prediction script.

```bash
python3 predict.py
```

or

```bash
python3 auto_predict.py
```

depending on the deployment configuration.

The script loads the trained `.pth` model, preprocesses the input image, performs inference, and displays the prediction.

---

# Tkinter Deployment Interface

A lightweight graphical user interface (GUI) was developed using **Tkinter** to simplify interaction with the deployed model.

The GUI performs the following tasks:

- Loads the trained `.pth` model.
- Allows the user to browse and select a skin lesion image.
- Captures images using the Raspberry Pi Camera.
- Preprocesses the input image.
- Performs model inference.
- Displays:
  - Predicted class (Benign / Malignant)
  - Confidence score
  - Clinical advisory message

> **Note:** The Tkinter GUI was used during deployment to demonstrate the system on the Raspberry Pi. The GUI source code is not included in this repository.

---

# Image Source Comparison

To evaluate the robustness of the deployed model, predictions were compared using two different image sources.

## 1. USB Image

High-quality reference images from the dataset were supplied to the Raspberry Pi using USB storage.

The trained model predicts:

- Benign / Malignant
- Prediction confidence

These results were considered the reference predictions.

---

## 2. Raspberry Pi Camera

The Raspberry Pi Camera Module was used to capture the same lesion under different lighting conditions.

For every captured image, the model predicts:

- Benign / Malignant
- Prediction confidence

---

# Comparison Parameters

The following metrics were compared:

- Predicted class
- Prediction confidence
- Agreement between USB and Raspberry Pi Camera predictions
- Effect of illumination on prediction confidence
- Image quality differences

This comparison helps evaluate how well the deployed model performs on real-world camera images compared with high-quality dataset images.

---

# Raspberry Pi Deployment Workflow

```
Train CNN Model
        │
        ▼
Save Trained Model (.pth)
        │
        ▼
Enable SSH & VNC
        │
        ▼
Connect Raspberry Pi and Laptop
        │
        ▼
Transfer Project using SCP
        │
        ▼
Install Dependencies
        │
        ▼
Load Trained Model
        │
        ▼
Capture Image (USB / Pi Camera)
        │
        ▼
Preprocess Image
        │
        ▼
Run Model Inference
        │
        ▼
Display Prediction
        │
        ▼
Compare Confidence Scores
```

---

# Deployment Interface

The deployed Raspberry Pi interface is shown below.

```markdown
![Raspberry Pi Deployment Interface](raspberry_pi_deployment_result_interface.png)
```

---

# Notes

- Ensure both the Raspberry Pi and laptop are connected to the same Wi-Fi network.
- Verify the Raspberry Pi IP address before connecting via VNC.
- Enable SSH before attempting file transfer with SCP.
- Place the trained `.pth` model inside the project directory before running inference.
- For best results, capture images under consistent lighting conditions to reduce prediction variability.

---

# Conclusion

The trained CNN model was successfully deployed on a Raspberry Pi 4/5 using PyTorch. Remote access was established using VNC Viewer, project files were transferred using SCP, and inference was performed locally on the Raspberry Pi. The deployment was further evaluated by comparing predictions from high-quality USB images and real-time Raspberry Pi Camera captures, demonstrating the feasibility of lightweight skin cancer detection on embedded hardware.
