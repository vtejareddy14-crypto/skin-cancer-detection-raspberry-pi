# 🔬 Skin Cancer Detection Using CNN on Raspberry Pi

> Real-time binary skin lesion classification (Benign vs Malignant) using a lightweight Depthwise Separable CNN deployed on Raspberry Pi 4 — no GPU required.

---

## 📌 Project Overview

This project implements a lightweight Convolutional Neural Network (DSC-CNN) for skin cancer detection, optimized for real-time inference on Raspberry Pi 4. Two training frameworks are compared — **PyTorch** and **Edge Impulse EON Tuner** — across 10, 20, and 30 training epochs.

The system outputs a clinician-advisory message with confidence score in under **1.3 seconds** per image on Raspberry Pi without any external GPU.

**Course:** BECE320E — Embedded C Programming  
**Institution:** Vellore Institute of Technology, Vellore  
**Team:** V. Sai Teja Reddy (23BVD0015) · L. Sai Varshit (23BEC0302)  
**Supervisor:** Dr. Manish Kumar

---

## 🎯 Key Results

| Framework | Epochs | Accuracy |
|---|---|---|
| PyTorch DSC-CNN | 10 | **92.3%** |
| Edge Impulse EON | 20 | **91.7%** |
| Proposed (CNN) | — | **91.2% @ 1.3s on Pi** |

- ✅ Real-time inference: **< 1.3s per image** on Raspberry Pi 4
- ✅ Model size after INT8 quantization: **< 8.5 MB**
- ✅ INT8 quantization: **74% size reduction**, only **0.4% accuracy drop**
- ✅ AUC-ROC: **0.91**

---

## 🧠 Model Architecture

Custom **DSC-CNN** (Depthwise Separable Convolution CNN) inspired by MobileNet:

```
Input (224×224×3)
    → Input Stem: 3×3 Conv + BN + ReLU6
    → 4× DSC Blocks (filters: 64, 128, 256, 256)
    → Global Average Pooling
    → Dropout (p=0.35)
    → Sigmoid Output (Benign / Malignant)
```

- **Parameters:** ~2.1 M
- **Model size (FP32):** ~8.5 MB
- **Computation reduction:** ~8.6× vs standard convolutions

---

## 📁 Repository Structure

```
CNN-PROJECT/
│
├── train_model.py              # Main PyTorch DSC-CNN training script
├── training_28.py              # Training variant (28-epoch run)
├── evaluate.py                 # Model evaluation — accuracy, F1, AUC
├── predict.py                  # Single image prediction (USB mode)
├── predict1.py                 # Prediction variant
├── auto_predict.py             # Automated batch prediction
├── cam.py                      # Live Pi Camera inference (Mode 2)
├── test.py                     # Test set evaluation script
│
├── convert.py                  # PyTorch → ONNX conversion
├── convert2.py                 # ONNX → TFLite conversion
├── tf_model.py                 # TensorFlow model definition
├── to_tflite.py                # TFLite export script
│
├── skin_cancer_model.pth       # Trained PyTorch model (base)
├── model_29_epoch.pth          # PyTorch model — 29 epochs
├── model20_epoch.pth           # PyTorch model — 20 epochs
├── model30_epoch_88.95_acc.pth # PyTorch model — 30 epochs (88.95%)
├── skin_cancer_model.onnx      # Exported ONNX model
├── edge_impulse_10_epochs_92.1_accuracy.eim  # Edge Impulse ARM binary
│
├── tf_model/                   # TensorFlow SavedModel directory
├── melanoma_cancer_dataset/    # HAM10000 dataset (local copy)
├── melanoma_cancer_dataset.zip # Dataset archive
└── captured.jpg                # Sample Pi Camera capture
```

---

## 📊 Dataset

**HAM10000** — Human Against Machine with 10000 training images

| Split | Images | Malignant | Benign |
|---|---|---|---|
| Train | 7,010 | 1,368 | 5,642 |
| Validation | 1,502 | 293 | 1,209 |
| Test | 1,503 | 293 | 1,210 |
| **Total** | **10,015** | **1,954** | **8,061** |

**Binary label mapping:**
- Benign → melanocytic nevi, benign keratoses, vascular lesions, dermatofibromas
- Malignant → melanoma, basal cell carcinoma, actinic keratoses

---

## ⚙️ Preprocessing Pipeline

1. Resize to **224×224** pixels
2. ImageNet normalization (mean = [0.485, 0.456, 0.406])
3. **Hair removal** — morphological black-hat filtering + inpainting (OpenCV Telea)
4. **CLAHE** — Contrast Limited Adaptive Histogram Equalization
5. Data augmentation — horizontal/vertical flips, ±15° rotation, zoom (0.9–1.1×), brightness jitter ±10%
6. Class-weighted oversampling — malignant weight = 4.0 (to handle 9:1 class imbalance)

---

## 🚀 Setup and Usage

### Requirements

```bash
pip install torch torchvision opencv-python pillow numpy scikit-learn
```

### Training

```bash
python train_model.py
```

### Evaluation

```bash
python evaluate.py
```

### Inference — USB Image Mode (Recommended)

```bash
python predict.py --image path/to/image.jpg
```

### Inference — Pi Camera Live Feed

```bash
python cam.py
```

### Convert to ONNX

```bash
python convert.py
```

---

## 🖥️ Raspberry Pi Deployment

**Hardware:**
- Raspberry Pi 4 (ARM Cortex-A72, quad-core, 4GB RAM)
- Raspberry Pi Camera Module v2 (Sony IMX219, 8MP)
- 7" HDMI Touchscreen

**Two input modes supported:**

| Mode | Input | SSIM | Confidence |
|---|---|---|---|
| Mode 1 (Recommended) | USB dermoscopic image | 1.00 (reference) | 88.5% |
| Mode 2 | Pi Camera live feed | 0.61 ± 0.09 | 74.3% |

**Advisory output displayed on Tkinter GUI:**
- 🔴 Malignant OR confidence < 75% → *"Consult a Dermatologist — Do not self-diagnose."*
- 🟢 Benign AND confidence ≥ 75% → *"Likely Normal — Consult a doctor for confirmation."*
- ⚠️ SSIM < 0.70 → *"Low image quality detected. Use a USB dermoscopic image."*

---

## 📈 Framework Comparison — PyTorch vs Edge Impulse

| Epochs | PyTorch Acc (%) | Edge Impulse Acc (%) |
|---|---|---|
| 10 | **92.3** | 91.2 |
| 20 | 88.77 | **91.7** |
| 30 | 86.3 | **92.4** |

**Key finding:** PyTorch converges faster at low epochs but overfits beyond epoch 10. Edge Impulse EON Tuner's hardware-aware regularization generalizes better at higher epochs — better suited for Raspberry Pi deployment.

---

## 🔧 Post-Training Optimization

| Technique | Effect |
|---|---|
| Structured Pruning | Removed bottom 25–30% filters per layer |
| INT8 Dynamic Quantization | 4× model size reduction, 2–3× latency reduction |
| ONNX Export (opset 13) | Cross-platform validation |
| TorchScript (.pt) | Raspberry Pi inference |
| Edge Impulse (.eim binary) | ARM-optimized deployment |

---

## 📋 Classification Metrics (Test Set)

| Metric | Benign | Malignant | Weighted Avg |
|---|---|---|---|
| Accuracy | 91.2% | 91.2% | **91.2%** |
| Precision | 91% | 91% | 91% |
| Recall | 93.1% | 89.3% | 91% |
| F1-Score | 91% | 91% | **91%** |
| AUC-ROC | — | — | **0.91** |

---

## 🏷️ Tech Stack

`Python` `PyTorch` `Edge Impulse` `OpenCV` `ONNX` `TFLite` `Raspberry Pi` `HAM10000` `Tkinter`

---

## 📄 License

This project was developed as part of academic coursework at VIT Vellore. For research and educational use only. Not intended for clinical diagnosis.
