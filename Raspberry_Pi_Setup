# Skin Cancer Detection using Edge Impulse on Raspberry Pi

## Overview

This project deploys a Skin Cancer Detection model trained using **Edge Impulse** on a **Raspberry Pi 4/5**. A **Tkinter-based GUI** allows users to select a skin lesion image and displays whether the lesion is **Benign** or **Malignant** along with the prediction confidence.

---

## Hardware Requirements

- Raspberry Pi 4 / Raspberry Pi 5
- MicroSD Card (16 GB or higher)
- Power Supply
- Laptop/Desktop
- Same Wi-Fi network for Raspberry Pi and Laptop

---

## Software Requirements

- Raspberry Pi OS
- Python 3
- Tkinter
- VNC Viewer
- SCP (Secure Copy)
- Edge Impulse Linux Deployment Package

---

## Required Python Packages

Install the required packages on Raspberry Pi.

```bash
sudo apt update
sudo apt install python3-pip python3-tk unzip

pip3 install numpy pillow edge_impulse_linux opencv-python
```

Or install using:

```bash
pip3 install -r requirements.txt
```

---

# Clone the Repository

```bash
git clone <repository-link>
cd <repository-folder>
```

---

# Download the Edge Impulse Model

1. Open your Edge Impulse Project.
2. Go to **Deployment**.
3. Select **Linux (ARM 64-bit)**.
4. Download the deployment package.
5. Extract the downloaded package.

Copy the generated model file (usually `.eim`) into the project folder.

Example:

```
model.eim
```

---

# Connecting Raspberry Pi to Laptop

Both Raspberry Pi and Laptop should be connected to the same Wi-Fi network.

### Find Raspberry Pi IP Address

```bash
hostname -I
```

Example:

```
192.168.1.20
```

---

## Enable VNC

Open Raspberry Pi Terminal.

```bash
sudo raspi-config
```

Navigate to

```
Interface Options
      ↓
VNC
      ↓
Enable
```

Restart Raspberry Pi if required.

---

## Connect Using VNC Viewer

Open **VNC Viewer** on your laptop.

Enter

```
RaspberryPi_IP_Address
```

Example

```
192.168.1.20
```

Login using Raspberry Pi credentials.

```
Username : <your_username>
Password : <your_password>
```

The Raspberry Pi desktop will now be accessible from your laptop.

---

# Transfer Project Files using SCP

Open terminal on your laptop.

Transfer an entire project folder.

```bash
scp -r SkinCancerProject <username>@<RaspberryPi_IP>:/home/<username>/
```

Example

```bash
scp -r SkinCancerProject teja-pi@192.168.1.20:/home/teja-pi/
```

Transfer a single file.

```bash
scp filename.py <username>@<RaspberryPi_IP>:/home/<username>/
```

---

# Navigate to Project Directory

```bash
cd ~/SkinCancerProject
```

---

# Project Structure

```
SkinCancerProject
│
├── model.eim
├── gui.py
├── inference.py
├── requirements.txt
├── assets/
├── images/
└── README.md
```

---

# Running the Application

Run the GUI.

```bash
python3 gui.py
```

---

# GUI Features

The Tkinter application provides:

- Browse and select a skin lesion image.
- Display the selected image.
- Run inference using the Edge Impulse model.
- Display prediction probabilities.
- Display final prediction:
  - Benign
  - Malignant

---

# Workflow

```
Train Model on Edge Impulse
            │
            ▼
Download Linux Deployment (.eim)
            │
            ▼
Transfer Project Files to Raspberry Pi using SCP
            │
            ▼
Connect Raspberry Pi using VNC Viewer
            │
            ▼
Run GUI Application
            │
            ▼
Select Skin Image
            │
            ▼
Edge Impulse Model Performs Inference
            │
            ▼
Display Prediction (Benign / Malignant)
```

---

# Troubleshooting

## Tkinter Module Not Found

Install Tkinter.

```bash
sudo apt install python3-tk
```

---

## SCP Permission Denied

Verify:

- Raspberry Pi is powered on.
- Both devices are connected to the same Wi-Fi.
- Username and IP address are correct.

---

## VNC Connection Failed

- Verify Raspberry Pi IP address.
- Ensure VNC is enabled.
- Confirm both devices are on the same network.

---

## Model File Not Found

Place the downloaded `.eim` file in the project directory before running the application.

---

# Technologies Used

- Python
- Edge Impulse
- Raspberry Pi 4 / Raspberry Pi 5
- Tkinter
- NumPy
- OpenCV
- Pillow
- VNC Viewer
- SCP

---

# Authors

Developed as part of an Edge AI-based Skin Cancer Detection project using Raspberry Pi and Edge Impulse.
