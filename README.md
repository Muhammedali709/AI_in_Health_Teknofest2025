# 🧠 AI-Powered Stroke Detection from CT Images (Teknofest 2025)

![Team](https://img.shields.io/badge/Team-KTU_MEDSIGHT_AI-blue?style=for-the-badge)
![Competition](https://img.shields.io/badge/Competition-Teknofest_2025-red?style=for-the-badge)
> **Institution:** Karadeniz Technical University (KTU)

## 📌 Project Overview
This project was developed for the **Teknofest 2025 "Artificial Intelligence in Health"** competition. Our goal was to develop a high-performance deep learning system to detect and segment **stroke (ischemic/hemorrhagic)** cases from raw Brain Computed Tomography (CT) images.

We engineered a **Hybrid Multi-Stage Architecture** that combines Object Detection (YOLO), Segmentation (Attention U-Net), and Anomaly Detection (GANs + ViT) to achieve state-of-the-art results.

## 🔬 Key Innovations (Özgünlük)
Our solution stands out due to three novel hybrid approaches designed by the **KTU MEDSIGHT AI** team:

1.  **Hybrid YOLO + Faster R-CNN:** Combines YOLO's speed for initial region proposal with Faster R-CNN's precision for detailed anomaly analysis.
2.  **YOLO + Attention U-Net Integration:** Uses YOLO for broad lesion localization and feeds those regions into an 8-layer Attention U-Net for pixel-perfect segmentation.
3.  **GAN-Based Anomaly Detection:** Utilizes a **Generative Adversarial Network (GAN)** to synthesize healthy brain data, followed by **Contrastive Learning** and **Vision Transformers (ViT)** to detect anomalies based on deviation from the "healthy" norm.

## 📊 Performance Benchmark
We rigorously tested **14 different architectures** on the Teknofest '21 and Brain CT datasets.

| Model Architecture | F1 Score (Validation) | Key Advantage |
|:-------------------|:--------------------:|:---------------------------|
| **Attention U-Net** | **0.93** | Best segmentation accuracy |
| **DenseNet201** | **0.93** | High feature retention |
| **EfficientNet B3** | 0.92 | Efficient parameter scaling |
| **ResNet101** | 0.91 | Deep residual learning |
| **YOLO v10** | 0.89 | Real-time detection speed |
| **Vision Transformer (ViT)** | 0.90 | Global context awareness |
| *Hybrid: YOLO + Attn U-Net* | 0.89 | Balanced speed/precision |

## 🛠️ Methodologies & Tech Stack
* **Object Detection:** YOLOv8, YOLOv10, Faster R-CNN
* **Segmentation:** U-Net, Attention U-Net, Mask R-CNN
* **Classification:** ResNet101, DenseNet121, EfficientNetB3
* **Generative AI:** GANs (for synthetic data augmentation)
* **Frameworks:** PyTorch, TensorFlow/Keras
* **Hardware:** Trained on NVIDIA **RTX 3090 (24GB VRAM)** + Intel i7-12700KF

## 📂 Project Structure
```text
AI_in_Health_Teknofest2025/
├── models/
│   ├── detection/       # YOLOv8 & Faster R-CNN scripts
│   ├── segmentation/    # Attention U-Net & Mask R-CNN
│   └── classification/  # EfficientNet & ViT implementations
├── data_augmentation/   # GAN & Contrastive Learning modules
├── web_interface/       # Deployment code for the diagnostic UI
├── reports/             # Project PDF and performance tables
└── README.md
```
## 📄 Official Project Report
For a deep dive into our architectural decisions, literature review, and full experimental results, please refer to our official submission report:

👉 [**View Full Project Presentation Report (PDF)**][220325_last.pdf](https://github.com/user-attachments/files/24773715/220325_last.pdf)


* **Document:** Teknofest 2025 AI in Health - Project Presentation Report
* **Language:** Turkish
* **Contents:**
  * Comprehensive Literature Review (YOLO, ResNet, ViT comparison)
  * Hybrid Architecture Diagrams (YOLO + Attention U-Net)
  * Detailed Error Analysis & GAN Implementation

## ⚠️ Note on Model Weights
Due to GitHub's file size limitations, the trained model weights (`.pt`, `.keras`, `.h5` files) are not included in this repository. 
* **Training Hardware:** Models were trained on an NVIDIA RTX 3090 (24GB VRAM) at the KTU AI Club Laboratory.

