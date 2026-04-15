

# ⚙️ PMSM Fault Detection using Deep Learning & ML

## 📌 Project Overview

This project implements a **real-time fault detection framework for a Permanent Magnet Synchronous Motor (PMSM)** using electrical signal data. The goal is to classify motor health conditions and detect faults early for predictive maintenance in industrial systems.

The system is trained on a large dataset of motor current signals and compares multiple machine learning approaches including **XGBoost and 1D CNNs**, along with feature engineering techniques like FFT and hybrid time-frequency representations.

---

## 📊 Dataset

* Source: Mendeley Dataset (PMSM motor signals)
* Total samples: **480,000 sliding windows**
* Window size: **256 time-steps**
* Input: Phase current signal (primarily `ai0`)
* Classes (4):

  * Healthy
  * Inter-turn fault
  * Inter-coil fault
  * Coil fault

---

## ⚙️ Feature Engineering

* Raw time-domain signal windows (256-length)
* FFT-based frequency features
* Hybrid time-frequency representations (experimental)
* Normalization and segmentation into fixed windows

---

## 🤖 Models Implemented

### 1. XGBoost Classifier

* Best accuracy: ~**74%**
* Strong baseline for structured features
* Limited performance on raw temporal dependencies

### 2. 1D Convolutional Neural Network (CNN)

* Input: (256 × 1) raw signal windows
* Architecture:

  * Conv1D + BatchNorm + MaxPooling layers
  * Global Average Pooling
  * Dense layers with Dropout
* Final model uses:

  * **Class weighting to handle imbalance**
  * Improved deeper architecture
* Best accuracy: ~**86%**

---

## 📈 Results Summary

* Overall best model: **Improved CNN (86% accuracy)**
* Strong class-wise performance:

  * Coil faults: ~100% detection accuracy
  * Healthy / Inter-turn / Inter-coil: moderate confusion
* Main challenge: distinguishing **inter-turn vs inter-coil faults**

---

## 🧠 Key Observations

* Deep learning (CNN) outperforms classical ML (XGBoost) for raw signal data
* Class imbalance significantly affects minority fault detection
* Time-domain CNN features outperform FFT-only approaches
* Model improves significantly with:

  * Class weights
  * Deeper convolutional architecture
  * Batch normalization

---

## 💾 Files in Repository

* `PMSM_X.npy`, `PMSM_y.npy` → dataset
* `cnn_model.keras` → trained CNN model
* `xgb_model.pkl` → trained XGBoost model
* Notebooks → preprocessing, training, evaluation

---

## 🚀 Future Work

* Real-time deployment using edge devices (Raspberry Pi / embedded system)
* Attention-based deep learning models (CNN-LSTM / Transformers)
* Improved separation of inter-turn and inter-coil faults
* Integration into full digital twin framework

---

## 📌 Conclusion

This project demonstrates a complete pipeline for PMSM fault detection using machine learning and deep learning techniques. The CNN-based approach shows strong potential for real-world predictive maintenance systems in industrial motor monitoring applications.

---
