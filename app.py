import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

from src.model import CNN
from src.dataset import CLASSES, get_inference_transforms

# 1. Device configuration
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"--> Web Prototype running on device: {device}")

# 2. Model initialization and weights loading
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "cifar10_cnn.pth")

model = CNN().to(device)
if os.path.exists(WEIGHTS_PATH):
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"--> Loaded trained checkpoint from: {WEIGHTS_PATH}")
else:
    print(f"[!] Warning: Checkpoint not found at {WEIGHTS_PATH}. Using untrained model.")

model.eval()

# 3. Transform pipeline
transform = get_inference_transforms()


def predict(input_image):
    """
    Receives an uploaded image, resizes to 32x32, normalizes,
    and returns class probabilities for the top predictions.
    """
    if input_image is None:
        return {}

    # Ensure PIL Image format
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    image = input_image.convert("RGB")

    # Preprocess
    tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Return dictionary of class names to probabilities
    return {CLASSES[i].capitalize(): float(probabilities[i]) for i in range(len(CLASSES))}


# 4. Gradio Interface Construction
title = "🧠 CIFAR-10 Image Classifier (PyTorch CNN)"
description = """
### Custom 3-Stage Convolutional Neural Network
Upload any photo to classify it into one of the 10 **CIFAR-10** categories:
*✈️ Airplane, 🚗 Automobile, 🐦 Bird, 🐱 Cat, 🦌 Deer, 🐶 Dog, 🐸 Frog, 🐴 Horse, 🚢 Ship, 🚚 Truck*

**Architecture Highlights:**
- **Feature Extractor**: 3 Conv blocks (32 → 64 → 128 channels) with `BatchNorm2d`, `ReLU`, `MaxPool2d(2, 2)`, and `Dropout2d(0.25)`.
- **Classification Head**: Dense layers with `Dropout(0.5)` to prevent overfitting.
- **Trained on**: 50,000 augmented CIFAR-10 images (~74.6% test accuracy).
"""

article = """
<div style="text-align: center; margin-top: 15px;">
    <p>Built with <b>PyTorch</b> and <b>Gradio</b> | Source code on <a href="https://github.com" target="_blank">GitHub</a></p>
</div>
"""

# Check for optional example images
example_dir = os.path.join(os.path.dirname(__file__), "examples")
examples = []
if os.path.exists(example_dir):
    examples = [
        os.path.join(example_dir, f)
        for f in sorted(os.listdir(example_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an Image or Take a Photo", sources=["upload", "clipboard", "webcam"]),
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions & Confidence"),
    title=title,
    description=description,
    article=article,
    examples=examples if examples else None,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
