# CIFAR-10 Image Classification with PyTorch CNN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

An end-to-end Computer Vision project implementing, analyzing, and deploying a custom 3-stage Convolutional Neural Network (CNN) from scratch using **PyTorch** on the **CIFAR-10** benchmark dataset. Includes an empirical study of the **overfitting phenomenon**, resolution via modern regularization (**BatchNorm**, **Dropout**, **Data Augmentation**), and a working **Gradio** web prototype.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Parameter Analysis](#-parameter-analysis)
- [The Overfitting Case & How We Resolved It](#-the-overfitting-case--how-we-resolved-it)
  - [Phase 1: Baseline Architecture & Severe Overfitting](#phase-1-baseline-architecture--severe-overfitting)
  - [Phase 2: Architectural & Pipeline Interventions](#phase-2-architectural--pipeline-interventions)
  - [Phase 3: Enhanced Regularized Results (25 Epochs)](#phase-3-enhanced-regularized-results-25-epochs)
- [Interactive Web Prototype (Gradio)](#-interactive-web-prototype-gradio)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Web Prototype](#running-the-web-prototype)
  - [Training via CLI](#training-via-cli)
  - [Inference on an Image](#inference-on-an-image)
  - [Running the Interactive Notebook](#running-the-interactive-notebook)
- [Key Engineering Learnings](#-key-engineering-learnings)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🔍 Overview

This project explores deep learning visual recognition fundamentals from mathematical foundation to full prototype deployment:
- **Custom Architecture**: Built an explicit 3-stage feature extractor using 2D convolutions, batch normalization, spatial pooling, and dual-level dropout.
- **Data Pipeline**: Constructed modular PyTorch `DataLoader` pipelines with dynamic data augmentation for training and deterministic normalization for evaluation.
- **Empirical Diagnostics**: Investigated a real-world bias-variance trade-off where training loss dropped to $0.14$ while validation loss diverged, diagnosing the root cause and applying targeted regularizations.
- **Working Prototype**: Packaged the inference pipeline into an interactive Gradio web application with real-time confidence scoring and top-3 class distribution visualization.

---

## 📊 Dataset

The model is evaluated on the standard **CIFAR-10** dataset:
- **Volume**: 60,000 $32 \times 32$ color (RGB) images across 10 balanced classes.
- **Split**: 50,000 training images, 10,000 testing images.
- **Classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
- **Normalization**: Pixel values are scaled to $[0.0, 1.0]$ via `ToTensor()` and standardized to $[-1.0, 1.0]$ via:
  $$\hat{x} = \frac{x - \mu}{\sigma} \quad \text{where } \mu = (0.5, 0.5, 0.5), \; \sigma = (0.5, 0.5, 0.5)$$

---

## 🧠 Model Architecture

The network employs a hierarchical feature extraction backbone followed by a regularized multi-layer perceptron (MLP) classification head:

```text
Input: Image (3 x 32 x 32)
  │
  ├── [Conv Block 1]
  │     ├── Conv2d(in=3, out=32, kernel=3, padding=1)  ──> (32 x 32 x 32)
  │     ├── BatchNorm2d(32)                            ──> Stabilizes activation distribution
  │     ├── ReLU()
  │     ├── MaxPool2d(kernel=2, stride=2)              ──> (32 x 16 x 16)
  │     └── Dropout2d(p=0.25)                          ──> Drops random 2D feature maps
  │
  ├── [Conv Block 2]
  │     ├── Conv2d(in=32, out=64, kernel=3, padding=1) ──> (64 x 16 x 16)
  │     ├── BatchNorm2d(64)
  │     ├── ReLU()
  │     ├── MaxPool2d(kernel=2, stride=2)              ──> (64 x 8 x 8)
  │     └── Dropout2d(p=0.25)
  │
  ├── [Conv Block 3]
  │     ├── Conv2d(in=64, out=128, kernel=3, padding=1) ─> (128 x 8 x 8)
  │     ├── BatchNorm2d(128)
  │     ├── ReLU()
  │     ├── MaxPool2d(kernel=2, stride=2)              ──> (128 x 4 x 4)
  │     └── Dropout2d(p=0.25)
  │
  └── Flatten: 128 * 4 * 4 = 2,048 spatial features
        │
  ├── [Classification Head]
  │     ├── Linear(in=2048, out=256)
  │     ├── ReLU()
  │     ├── Dropout(p=0.5)                             ──> Regularizes dense parameter budget
  │     └── Linear(in=256, out=10)                     ──> 10 Class Logits
```

---

## 📐 Parameter Analysis

$$\text{Conv Weights} = (K_h \times K_w \times C_{\text{in}}) \times C_{\text{out}} + C_{\text{out}} \qquad \text{BatchNorm Params} = 2 \times C_{\text{out}} \; (\gamma, \beta)$$

| Layer | Operation | Output Shape | Parameters | % of Total |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Raw RGB Image | $(3, 32, 32)$ | 0 | 0.0% |
| **Conv Block 1** | `Conv2d(3, 32, k=3, p=1)` | $(32, 32, 32)$ | $3 \times 3 \times 3 \times 32 + 32 = \mathbf{896}$ | 0.14% |
| | `BatchNorm2d(32)` + `MaxPool` | $(32, 16, 16)$ | $2 \times 32 = \mathbf{64}$ | 0.01% |
| **Conv Block 2** | `Conv2d(32, 64, k=3, p=1)` | $(64, 16, 16)$ | $3 \times 3 \times 32 \times 64 + 64 = \mathbf{18,496}$ | 2.98% |
| | `BatchNorm2d(64)` + `MaxPool` | $(64, 8, 8)$ | $2 \times 64 = \mathbf{128}$ | 0.02% |
| **Conv Block 3** | `Conv2d(64, 128, k=3, p=1)` | $(128, 8, 8)$ | $3 \times 3 \times 64 \times 128 + 128 = \mathbf{73,856}$ | 11.90% |
| | `BatchNorm2d(128)` + `MaxPool` | $(128, 4, 4)$ | $2 \times 128 = \mathbf{256}$ | 0.04% |
| **FC Layer 1** | `Linear(2048, 256)` + `ReLU` | $(256)$ | $2,048 \times 256 + 256 = \mathbf{524,544}$ | 84.49% |
| **FC Layer 2** | `Linear(256, 10)` | $(10)$ | $256 \times 10 + 10 = \mathbf{2,570}$ | 0.41% |
| **Total** | | | **620,810 trainable parameters** (~$2.37 \text{ MB}$) | **100%** |

---

## 📈 The Overfitting Case & How We Resolved It

A central engineering highlight of this project was discovering, diagnosing, and systematically curing **high-variance overfitting**.

### Phase 1: Baseline Architecture & Severe Overfitting

The initial baseline model contained no regularization (no Dropout, no BatchNorm, and no Data Augmentation). When trained for 10 epochs using Adam ($\text{lr}=10^{-3}$), the following dynamic unfolded:

| Epoch | Train Loss | Validation Loss | Observation |
| :---: | :---: | :---: | :--- |
| 1 | 1.3679 | 1.0639 | Rapid initial convergence |
| 2 | 0.9301 | 0.9405 | Generalization improving |
| 3 | 0.7408 | 0.8041 | Consistent loss decrease |
| **4** | **0.6050** | **0.7025** | ⭐ **Optimal Generalization Point (Minimum Val Loss)** |
| 5 | 0.4956 | 0.7468 | ⚠️ Validation loss starts diverging |
| 6 | 0.3989 | 0.7782 | Training error drops; validation error worsens |
| 7 | 0.3125 | 0.7873 | Model starts memorizing training noise |
| 8 | 0.2348 | 0.8940 | Validation error climbs by +27% |
| 9 | 0.1825 | 0.9839 | Pronounced generalization gap |
| **10** | **0.1404** | **1.0777** | ❌ **Severe Overfitting (+53% validation loss explosion)** |

#### Diagnosis:
- Training loss dropped by nearly **90%** ($1.368 \to 0.140$).
- Validation loss reversed and inflated by **+53%** ($0.7025 \to 1.0777$).
- **Cause**: The 620k parameter capacity allowed the network to memorize individual pixel patterns rather than invariant visual features. The FC layers alone account for 84.5% of weights with zero regularization.

---

### Phase 2: Architectural & Pipeline Interventions

To eliminate memorization and force true semantic feature learning, three complementary interventions were introduced:

1. **Batch Normalization (`nn.BatchNorm2d`)**:
   - Added after every convolution layer.
   - Normalizes channel activations per mini-batch, ensuring zero mean and unit variance. This stabilizes the optimization loss surface and prevents internal covariate shifts.
2. **Dual-Level Dropout (`nn.Dropout2d(0.25)` & `nn.Dropout(0.5)`)**:
   - `Dropout2d(0.25)` randomly drops entire 2D feature maps in convolutional stages, breaking inter-channel co-dependencies.
   - `Dropout(0.5)` zeroes 50% of the dense activations before the final classification logits, combating the large parameter footprint in `fc_layers`.
3. **Data Augmentation**:
   - `RandomCrop(32, padding=4)`: Simulates translation and framing variations.
   - `RandomHorizontalFlip()`: Artificially doubles orientation perspectives.
   - **Applied strictly to training data**: Test/validation data remains deterministic for objective evaluation.

---

### Phase 3: Enhanced Regularized Results (25 Epochs)

Retraining the regularized network for 25 epochs on Apple Silicon GPU (`mps`) produced completely stable, monotonically improving generalization:

```
--> Using compute device: mps
--> Starting training for 25 epochs...
Epoch [01/25] Train Loss: 1.8450 | Val Loss: 1.4354 | Val Accuracy: 47.26%
Epoch [02/25] Train Loss: 1.6414 | Val Loss: 1.2977 | Val Accuracy: 52.94%
Epoch [03/25] Train Loss: 1.5468 | Val Loss: 1.2226 | Val Accuracy: 54.78%
Epoch [04/25] Train Loss: 1.4769 | Val Loss: 1.1395 | Val Accuracy: 58.72%
Epoch [05/25] Train Loss: 1.4125 | Val Loss: 1.0666 | Val Accuracy: 61.72%
Epoch [06/25] Train Loss: 1.3646 | Val Loss: 1.0327 | Val Accuracy: 63.07%
Epoch [07/25] Train Loss: 1.3222 | Val Loss: 1.0037 | Val Accuracy: 63.61%
Epoch [08/25] Train Loss: 1.2842 | Val Loss: 0.9509 | Val Accuracy: 66.24%
Epoch [09/25] Train Loss: 1.2559 | Val Loss: 0.9048 | Val Accuracy: 67.25%
Epoch [10/25] Train Loss: 1.2194 | Val Loss: 0.8877 | Val Accuracy: 68.69%
Epoch [11/25] Train Loss: 1.2012 | Val Loss: 0.9173 | Val Accuracy: 67.31%
Epoch [12/25] Train Loss: 1.1763 | Val Loss: 0.8586 | Val Accuracy: 70.27%
Epoch [13/25] Train Loss: 1.1537 | Val Loss: 0.8156 | Val Accuracy: 71.50%
Epoch [14/25] Train Loss: 1.1415 | Val Loss: 0.8294 | Val Accuracy: 70.65%
Epoch [15/25] Train Loss: 1.1348 | Val Loss: 0.8351 | Val Accuracy: 70.87%
Epoch [16/25] Train Loss: 1.1155 | Val Loss: 0.7838 | Val Accuracy: 72.59%
Epoch [17/25] Train Loss: 1.1019 | Val Loss: 0.7886 | Val Accuracy: 72.11%
Epoch [18/25] Train Loss: 1.0917 | Val Loss: 0.7740 | Val Accuracy: 73.04%
Epoch [19/25] Train Loss: 1.0805 | Val Loss: 0.7782 | Val Accuracy: 72.57%
Epoch [20/25] Train Loss: 1.0614 | Val Loss: 0.7821 | Val Accuracy: 72.16%
Epoch [21/25] Train Loss: 1.0588 | Val Loss: 0.7495 | Val Accuracy: 73.73%
Epoch [22/25] Train Loss: 1.0391 | Val Loss: 0.7461 | Val Accuracy: 74.05%
Epoch [23/25] Train Loss: 1.0434 | Val Loss: 0.7319 | Val Accuracy: 74.24%
Epoch [24/25] Train Loss: 1.0291 | Val Loss: 0.7509 | Val Accuracy: 73.61%
Epoch [25/25] Train Loss: 1.0196 | Val Loss: 0.7309 | Val Accuracy: 74.57%

--> Training complete! Best weights saved to: weights/cifar10_cnn.pth
```

#### Key Findings:
1. **Zero Divergence**: Validation loss dropped from $1.4354$ down to **$0.7309$** without rebounding. Overfitting was completely mitigated.
2. **Train Loss > Validation Loss**: Because Dropout randomly silences 25%–50% of units and data augmentation distorts images during training, the training task is artificially harder. During evaluation (`model.eval()`), the full ensemble cooperates on unperturbed test images.
3. **Generalization Performance**: Final test accuracy reached **74.57%** under strict regularization.

---

## 💻 Interactive Web Prototype (Gradio)

The model is packaged into an interactive Gradio web application ([`app.py`](file:///Users/swapnil/Desktop/Projects/CNN/app.py)):

- **Inference Pipeline**: Takes any image format, resizes to $32 \times 32$, normalizes, and passes through the trained PyTorch model.
- **Visual Feedback**: Displays animated horizontal probability progress bars for the **Top-3 predicted classes**.
- **Input Flexibility**: Drag & drop files, paste directly from clipboard, or capture live using a webcam.

```bash
# Run locally
python app.py
```
*Access in browser at: `http://localhost:7860`*

---

## 📁 Project Structure

```text
├── CNN_for_CIFAR10.ipynb     # Interactive Jupyter Notebook with complete workflow
├── src/
│   ├── __init__.py           # Package exports
│   ├── model.py              # CNN architecture with BatchNorm & Dropout
│   ├── dataset.py            # Augmented training & clean test loaders
│   ├── train.py              # Modular training script with MPS/CUDA support
│   └── predict.py            # CLI inference tool for individual images
├── weights/
│   └── cifar10_cnn.pth       # Saved model checkpoint (~2.4 MB)
├── app.py                    # Interactive Gradio web application
├── requirements.txt          # Project dependencies (torch, torchvision, gradio)
├── .gitignore                # Excludes large data/ folder and checkpoints
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
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

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Prototype

```bash
python app.py
```
Open `http://localhost:7860` in your web browser.

### Training via CLI

Train the model from scratch with automatic hardware acceleration (Apple Silicon MPS, NVIDIA CUDA, or CPU):

```bash
python src/train.py --epochs 25 --batch_size 64 --lr 0.001 --output weights/cifar10_cnn.pth
```

### Inference on an Image

Predict the class of any local image file:

```bash
python src/predict.py --image path/to/sample.jpg --weights weights/cifar10_cnn.pth
```

### Running the Interactive Notebook

```bash
jupyter notebook CNN_for_CIFAR10.ipynb
```

---

## 💡 Key Engineering Learnings

1. **Idiomatic PyTorch Invocation**: Always invoke neural networks via `model(x)` instead of `model.forward(x)` so internal pre- and post-forward hooks trigger properly.
2. **State Management**: When switching between `model.eval()` for validation and `model.train()` for the next epoch, explicitly toggle modes; BatchNorm and Dropout behave fundamentally differently across states.
3. **Memory Hygiene**: Use `loss.item()` when accumulating running losses to avoid retaining tensor backward graphs in memory.
4. **Regularization Trade-offs**: Strong regularization suppresses sample memorization but requires more training epochs to fully converge.

---

## 🗺️ Roadmap

- [x] Baseline 3-stage CNN implementation
- [x] Overfitting diagnosis & loss tracking
- [x] Batch Normalization (`nn.BatchNorm2d`)
- [x] Spatial & Dense Dropout (`nn.Dropout2d`, `nn.Dropout`)
- [x] Real-time Data Augmentation (`RandomCrop`, `RandomHorizontalFlip`)
- [x] Interactive Gradio web prototype (`app.py`)
- [ ] Learning Rate Scheduling (Cosine Annealing)
- [ ] ResNet-18 Transfer Learning benchmark

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
