# CIFAR-10 Image Classification with PyTorch CNN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

A custom Convolutional Neural Network (CNN) built and trained from scratch using PyTorch on the **CIFAR-10** benchmark dataset, reaching **75.61% test accuracy**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Parameter Analysis](#-parameter-analysis)
- [Training Dynamics & Overfitting Analysis](#-training-dynamics--overfitting-analysis)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training via CLI](#training-via-cli)
  - [Inference on an Image](#inference-on-an-image)
  - [Running the Notebook](#running-the-notebook)
- [Key Engineering Learnings](#-key-engineering-learnings)
- [Roadmap (v2.0 Improvements)](#-roadmap-v20-improvements)
- [License](#-license)

---

## 🔍 Overview

This project explores the fundamentals of Deep Learning computer vision by designing, implementing, evaluating, and diagnosing a 3-stage Convolutional Neural Network from scratch.

### Key Objectives:
- Implement custom 2D convolution blocks with non-linear activation (ReLU) and spatial pooling (MaxPool2d).
- Build a modular data ingestion pipeline using `torchvision.transforms` and PyTorch `DataLoader`.
- Track loss progression across epochs and analyze the empirical onset of overfitting.
- Export trained state dictionaries (`.pth`) for deployment and downstream inference.

---

## 📊 Dataset

The model is trained on **CIFAR-10** (Canadian Institute for Advanced Research, 10 classes):
- **Volume**: 60,000 $32 \times 32$ color (RGB) images.
- **Split**: 50,000 training images, 10,000 testing images.
- **Classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
- **Normalization**: Pixel values are scaled to $[0.0, 1.0]$ via `ToTensor()` and standardized to $[-1.0, 1.0]$ via:
  $$\hat{x} = \frac{x - \mu}{\sigma} \quad \text{where } \mu = 0.5, \; \sigma = 0.5$$

---

## 🧠 Model Architecture

The network consists of three hierarchical convolutional stages that progressively extract richer spatial features while downsampling spatial dimensions, followed by a multi-layer perceptron (MLP) classification head:

```text
Input: Image (3 x 32 x 32)
  │
  ├──> Conv2d(in=3, out=32, kernel=3, padding=1) ──> (32 x 32 x 32)
  ├──> ReLU()
  └──> MaxPool2d(kernel=2, stride=2)              ──> (32 x 16 x 16)
  │
  ├──> Conv2d(in=32, out=64, kernel=3, padding=1) ──> (64 x 16 x 16)
  ├──> ReLU()
  └──> MaxPool2d(kernel=2, stride=2)              ──> (64 x 8 x 8)
  │
  ├──> Conv2d(in=64, out=128, kernel=3, padding=1) ─> (128 x 8 x 8)
  ├──> ReLU()
  └──> MaxPool2d(kernel=2, stride=2)              ──> (128 x 4 x 4)
  │
  └──> Flatten: 128 * 4 * 4 = 2,048 features
  │
  ├──> Linear(in=2048, out=256)
  ├──> ReLU()
  └──> Linear(in=256, out=10)                     ──> 10 Class Logits
```

---

## 📐 Parameter Analysis

$$\text{Conv Layer Weights} = (K_h \times K_w \times C_{\text{in}}) \times C_{\text{out}} + C_{\text{out}}$$

| Layer | Operation | Output Shape | Parameters | % of Total |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Raw RGB Image | $(3, 32, 32)$ | 0 | 0.0% |
| **Conv Block 1** | `Conv2d(3, 32, k=3, p=1)` + `MaxPool` | $(32, 16, 16)$ | $3 \times 3 \times 3 \times 32 + 32 = \mathbf{896}$ | 0.14% |
| **Conv Block 2** | `Conv2d(32, 64, k=3, p=1)` + `MaxPool` | $(64, 8, 8)$ | $3 \times 3 \times 32 \times 64 + 64 = \mathbf{18,496}$ | 2.98% |
| **Conv Block 3** | `Conv2d(64, 128, k=3, p=1)` + `MaxPool` | $(128, 4, 4)$ | $3 \times 3 \times 64 \times 128 + 128 = \mathbf{73,856}$ | 11.91% |
| **FC Layer 1** | `Linear(2048, 256)` + `ReLU` | $(256)$ | $2,048 \times 256 + 256 = \mathbf{524,544}$ | 84.55% |
| **FC Layer 2** | `Linear(256, 10)` | $(10)$ | $256 \times 10 + 10 = \mathbf{2,570}$ | 0.41% |
| **Total** | | | **620,362 params** (~$2.37 \text{ MB}$) | **100%** |

> **Note**: The first fully-connected layer contains **84.55%** of the entire parameter budget, which is typical for classical CNN architectures without Global Average Pooling (GAP).

---

## 📈 Training Dynamics & Overfitting Analysis

### Hyperparameters
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.Adam` ($\beta_1=0.9, \beta_2=0.999$, default $\text{lr}=10^{-3}$)
- **Batch Size**: 64
- **Epochs**: 10

### Loss Progression

| Epoch | Train Loss | Validation Loss | Observation |
| :---: | :---: | :---: | :--- |
| 1 | 1.3679 | 1.0639 | Rapid initial convergence |
| 2 | 0.9301 | 0.9405 | Steady generalization improvement |
| 3 | 0.7408 | 0.8041 | Consistent downward trajectory |
| **4** | **0.6050** | **0.7025** | ⭐ **Global Minimum Validation Loss (Optimal Generalization)** |
| 5 | 0.4956 | 0.7468 | ⚠️ Validation loss starts diverging |
| 6 | 0.3989 | 0.7782 | Training continues to drop; validation worsens |
| 7 | 0.3125 | 0.7873 | Model begins memorizing training noise |
| 8 | 0.2348 | 0.8940 | Validation error increases by +27% over best epoch |
| 9 | 0.1825 | 0.9839 | Pronounced divergence between curves |
| 10 | 0.1404 | 1.0777 | Overfitting confirmed (+53% validation loss climb) |

**Final Test Accuracy**: **75.61%**

### Engineering Diagnosis
The network displays textbook **high variance (overfitting)**:
- Training loss dropped by nearly **90%** ($1.368 \to 0.140$).
- Validation loss bottomed at Epoch 4 ($0.7025$) and rebounded by **+53%** ($1.0777$) by Epoch 10.
- **Root Cause**: The network capacity (~620k weights) exceeds the information entropy without explicit regularization (no Dropout, no weight decay, no BatchNorm, and no data augmentation).

---

## 📁 Project Structure

```text
├── CNN_for_CIFAR10.ipynb     # Interactive Jupyter Notebook with complete workflow
├── src/
│   ├── __init__.py           # Package exports
│   ├── model.py              # CNN architecture class (nn.Module)
│   ├── dataset.py            # DataLoader helpers, labels, and transforms
│   ├── train.py              # Standalone training script with validation & checkpointing
│   └── predict.py            # Single-image inference CLI
├── weights/
│   └── cifar10_cnn.pth       # Saved model checkpoint (~2.4 MB)
├── requirements.txt          # Python environment dependencies
├── .gitignore                # Excludes large data/ folder and checkpoints
├── LICENSE                   # MIT License
└── README.md                 # Complete documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CIFAR10-CNN-Classifier.git
   cd CIFAR10-CNN-Classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Training via CLI

Run the modular training script with automatic device acceleration (CUDA, Apple Silicon MPS, or CPU):

```bash
python src/train.py --epochs 10 --batch_size 64 --lr 0.001
```

### Inference on an Image

Predict the class of any local image:

```bash
python src/predict.py --image path/to/sample.jpg --weights weights/cifar10_cnn.pth
```

### Running the Notebook

If you prefer exploring interactively:

```bash
jupyter notebook CNN_for_CIFAR10.ipynb
```

---

## 💡 Key Engineering Learnings

1. **PyTorch Idioms**: Always invoke models via `model(x)` instead of `model.forward(x)` to ensure register hooks and profiler events trigger correctly.
2. **State Management**: Calling `model.eval()` before validation requires explicitly toggling `model.train()` back at the start of each training epoch.
3. **Memory Management**: Accumulate losses with `loss.item()` to avoid keeping tensor computation graphs in memory.
4. **Generalization Gap**: Monitoring validation loss at every epoch is crucial—stopping training at Epoch 4 (Early Stopping) would have yielded the best generalizing checkpoint.

---

## 🗺️ Roadmap (v2.0 Improvements)

- [ ] **Batch Normalization**: Add `nn.BatchNorm2d` after each convolution to stabilize activation distributions.
- [ ] **Dropout**: Inject `nn.Dropout2d(0.25)` and `nn.Dropout(0.5)` to suppress feature co-adaptation.
- [ ] **Data Augmentation**: Apply `RandomHorizontalFlip()` and `RandomCrop(32, padding=4)` to artificially broaden training variety.
- [ ] **Interactive Prototype**: Deploy on Hugging Face Spaces using Gradio for real-time web testing.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
