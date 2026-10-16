import argparse
import os
import torch
from PIL import Image

try:
    from .model import CNN
    from .dataset import CLASSES, get_inference_transforms
except ImportError:
    from model import CNN
    from dataset import CLASSES, get_inference_transforms


def get_device() -> torch.device:
    """Detect and return the best available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights_path: str = "weights/cifar10_cnn.pth", device: torch.device = None) -> CNN:
    """Loads trained weights into CNN model and sets eval mode."""
    if device is None:
        device = get_device()

    model = CNN().to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(image_path: str, weights_path: str = "weights/cifar10_cnn.pth", top_k: int = 3):
    """Predicts class probabilities for an input image."""
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(weights_path, device)
    transform = get_inference_transforms()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, top_k)

    print(f"\nPredictions for '{image_path}':")
    print("-" * 40)
    for prob, idx in zip(top_probs, top_indices):
        class_name = CLASSES[idx.item()]
        pct = prob.item() * 100
        print(f"  {class_name.capitalize():<12}: {pct:6.2f}%")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Single Image Prediction CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--weights", type=str, default="weights/cifar10_cnn.pth", help="Path to .pth weights")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")
    args = parser.parse_args()

    predict(args.image, args.weights, args.top_k)
