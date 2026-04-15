

# 📊 PMSM Fault Detection using CNN & XGBoost

## 🧠 Project Overview

This project builds a **real-time digital twin framework for PMSM (Permanent Magnet Synchronous Motor) fault detection** using machine learning and deep learning. The goal is to classify motor health conditions from raw electrical current signals and enable predictive maintenance.

The dataset consists of **480,000 sliding windows (length = 256 samples)** derived from phase current signals collected under different operating conditions (speed, load, and fault scenarios).

---

## ⚙️ Motor Conditions & Classes

The model classifies motor states into **4 categories**:

* **0 → Healthy motor**
* **1 → Interturn fault**
* **2 → Intercoil fault**
* **3 → Coil fault**

---

## 📦 Dataset

* Source: [https://data.mendeley.com/datasets/rgn5brrgrn/5](https://data.mendeley.com/datasets/rgn5brrgrn/5)
* Input: Phase current signal (AI0 channel from TDMS data)
* Shape: `(480,000 samples → 256-point windows)`
* Preprocessing: normalization + windowing + optional FFT/hybrid features

---

## 🏗️ Models Used

### 1. XGBoost (Baseline ML Model)

* Handcrafted + FFT-based features
* Accuracy: ~74%
* Strong baseline but limited temporal understanding

### 2. 1D Convolutional Neural Network (Final Model)

* Input: raw 256-length signal windows

* Architecture:

  * Conv1D → BatchNorm → MaxPooling (x2)
  * Conv1D (256 filters)
  * GlobalAveragePooling
  * Dense + Dropout
  * Softmax (4 classes)

* Training improvements:

  * Class weighting (to handle imbalance)
  * Adam optimizer (lr = 0.0005)
  * Early stopping + best model checkpoint

* Final Accuracy: **~86%**

---

## 📈 Key Results

| Model       | Accuracy | Notes                 |
| ----------- | -------- | --------------------- |
| XGBoost     | ~74%     | Feature-based         |
| CNN (v1)    | ~71%     | Initial architecture  |
| CNN (final) | **~86%** | Balanced + deeper CNN |

### Key Observations

* Coil faults are easiest to detect (near-perfect recall)
* Interturn vs intercoil confusion remains the hardest challenge
* CNN significantly outperforms classical ML when trained on raw sequences

---

## 📉 Confusion Behavior

* Strong performance on **Healthy & Coil faults**
* Most misclassification occurs between:

  * Interturn ↔ Intercoil faults

---

## 💾 Saved Models

* `cnn_model.keras` → trained deep learning model
* `xgb_model.pkl` → trained XGBoost model
* Best CNN checkpoint saved during training via ModelCheckpoint

---

## 🚀 How to Run

1. Load dataset (`.npy` files)
2. Reshape into `(samples, 256, 1)`
3. Load saved models from Google Drive
4. Run inference directly without retraining

---

## 🔮 Future Improvements

* Add LSTM/Transformer-based temporal modeling
* Improve separation of interturn vs intercoil faults
* Deploy as real-time monitoring system (Raspberry Pi / edge device)
* Expand to multi-sensor IMU + current fusion

---

## 📌 Summary

This project demonstrates a complete pipeline for **PMSM fault detection using deep learning**, progressing from feature-based ML models to a robust CNN-based architecture capable of learning directly from raw motor current signals.

---
