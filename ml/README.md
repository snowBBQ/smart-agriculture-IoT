# 🍇 Grape Leaf Disease Detection with YOLOv8 & ONNX Runtime

An end-to-end Computer Vision pipeline for real-time detection and classification of grape leaf diseases using **YOLOv8 Nano** and lightweight **ONNX Runtime** inference.

---

## 📌 Features

- **High Precision & Speed**: Powered by `yolov8n`, optimized for fast edge/CPU/GPU deployment.
- **Pure ONNX Runtime Inference**: Zero PyTorch dependency required during production inference.
- **Class-Aware NMS**: Built-in coordinate offset trick to prevent multi-class box suppression on overlapping leaves.
- **Hardened Against False Positives**: Retrained with **Negative Background Samples** to eliminate background hallucinations on non-leaf textures, background shadows, and greenhouse elements.
- **Letterbox Preprocessing**: Preserves original image aspect ratios during resizing.

---

## 📊 Model Performance & Metrics

The model was trained for 100 epochs using **YOLOv8 Nano** at `imgsz=640`. Incorporating negative background samples reduced false positive predictions on background regions by **over 70%**.

| Metric | Score | Description |
| :--- | :---: | :--- |
| **mAP@50** | **`0.980`** | Mean Average Precision at IoU threshold 0.50 |
| **mAP@50-95** | **`0.950`** | Mean Average Precision across IoU thresholds 0.50 to 0.95 |
| **Precision (B)** | **`0.955`** | Fraction of correct positive detections |
| **Recall (B)** | **`0.960`** | Fraction of total actual objects detected |

### Normalized Class Accuracy (Confusion Matrix)

| Class Name | Accuracy |
| :--- | :---: |
| `Grape__Esca` | **100%** |
| `Grape__LeafBlight` | **100%** |
| `Grape__BlackRot` | **95%** |
| `Grape__Healthy` | **95%** |

---

## 📁 Dataset Overview

The dataset is based on the [Grape Leaf Diseases Dataset](https://www.kaggle.com/datasets/yusufmurtaza01/grape-leaf-diseases) from Kaggle, enhanced with custom background samples.

### Target Classes (4 Classes)
- **`Grape__BlackRot`**: Fungal infection causing dark circular spots.
- **`Grape__Esca`**: Complex disease causing tiger-stripe discoloration patterns.
- **`Grape__Healthy`**: Healthy grape leaf without visible pathologies.
- **`Grape__LeafBlight`**: Lesions leading to leaf drying and tissue death.

### Background Hardening (Negative Samples)
- **Train Set**: 1,040 images (including 53 background samples without bounding boxes).
- **Validation Set**: 210 images (including 29 background samples).
- **Impact**: Reduced false positives on non-leaf objects (`background -> Healthy`) from **77** to **23** instances at default confidence thresholds.