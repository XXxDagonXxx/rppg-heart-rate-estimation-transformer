#🫀 rppg-heart-rate-estimation-transformer


> Contactless heart rate estimation from facial videos using Computer Vision, Signal Processing, and Deep Learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-red)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📌 Overview

This project implements an end-to-end **Remote Photoplethysmography (rPPG)** system capable of estimating heart rate directly from facial videos without requiring any physical sensor.

The framework combines classical physiological signal processing with a modern Dual-Stream CNN-Transformer architecture to learn robust temporal representations of subtle skin color variations caused by blood circulation.

The complete pipeline includes

- Face Detection
- Face Landmark Tracking
- ROI Extraction
- Skin Segmentation
- Signal Enhancement
- Deep Learning
- Heart Rate Prediction

---

## 🎯 Features

- Contactless Heart Rate Estimation
- MediaPipe FaceMesh based facial landmark tracking
- Forehead and Cheek ROI extraction
- Skin Mask Generation
- CLAHE Image Enhancement
- POS Signal Extraction
- CHROM Signal Extraction
- Butterworth Bandpass Filtering
- Sliding Window Dataset Generation
- Multi-Scale CNN Feature Encoder
- Dual-Stream CNN + Transformer Network
- CBAM Attention
- Spectral Attention
- Mixed Precision Training
- EMA Weight Averaging
- Automatic Evaluation Pipeline
- Deployment Ready Inference

---

## 🏗 Project Pipeline

```
Facial Video
      │
      ▼
MediaPipe FaceMesh
      │
      ▼
ROI Extraction
(Forehead + Cheeks)
      │
      ▼
Skin Segmentation
      │
      ▼
CLAHE Enhancement
      │
      ▼
Signal Processing
(POS + CHROM)
      │
      ▼
Sliding Window Creation
      │
      ▼
Dual Stream CNN
      │
      ▼
Transformer Encoder
      │
      ▼
Heart Rate Prediction
```

---

## 🧠 Model Architecture

The proposed model consists of:

### Appearance Stream

Extracts spatial texture and color information from facial ROIs using a Multi-Scale CNN encoder.

### Motion Stream

Captures temporal motion between consecutive frames to enhance pulse signal learning.

### Transformer Encoder

Learns long-range temporal dependencies across the input sequence.

### Prediction Head

Outputs the estimated heart rate.

---

## 📊 Signal Processing Pipeline

The preprocessing pipeline includes:

- Face Detection
- Facial Landmark Localization
- ROI Selection
- Skin Pixel Segmentation
- Green Channel Extraction
- POS Algorithm
- CHROM Algorithm
- Detrending
- Bandpass Filtering
- Signal Normalization

---

## 📚 Dataset

Dataset used:

**UBFC-rPPG Dataset**

Contains synchronized facial videos and physiological ground-truth signals.

---

## 📈 Training

Training includes

- Mixed Precision Training
- AdamW Optimizer
- Learning Rate Scheduler
- EMA Model Averaging
- Gradient Clipping
- Early Stopping
- Automatic Checkpoint Saving

---

## 📊 Evaluation Metrics

Performance is evaluated using

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation
- Signal-to-Noise Ratio (SNR)

---

## 🚀 Inference

The trained model can estimate heart rate directly from an unseen facial video.

```
Input Video
      │
      ▼
Preprocessing
      │
      ▼
Model Prediction
      │
      ▼
Estimated Heart Rate
```

---

## 💻 Installation

```bash
git clone https://github.com/<your-username>/rppg-heart-rate-estimation-transformer.git

cd rppg-heart-rate-estimation-transformer

pip install -r requirements.txt
```

---

## ▶️ Training

```bash
python train.py
```

---

## ▶️ Evaluation

```bash
python evaluate.py
```

---

## ▶️ Inference

```bash
python inference.py
```

---

## 🛠 Technologies Used

- Python
- TensorFlow
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Matplotlib
- Scikit-Learn

---

## 🔬 Future Improvements

- Real-time Webcam Inference
- Respiratory Rate Estimation
- SpO₂ Estimation
- Mobile Deployment
- ONNX Export
- TensorRT Optimization
- Multi-person Heart Rate Estimation
- Explainable AI Visualizations

---

## 📄 License

MIT License

---

## ⭐ Acknowledgements

- UBFC-rPPG Dataset
- TensorFlow
- MediaPipe
- OpenCV
