import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from typing import Tuple

# CIFAR-10 10 Class Labels in order (0 to 9)
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def get_train_transforms() -> transforms.Compose:
    """
    Returns training transforms with Data Augmentation:
    - RandomCrop(32, padding=4): simulates shifts and framing variations
    - RandomHorizontalFlip(): doubles perspective orientations
    - ToTensor(): scales pixels from [0, 255] to [0.0, 1.0]
    - Normalize: centers to [-1.0, 1.0]
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_test_transforms() -> transforms.Compose:
    """
    Returns deterministic transforms for evaluation/testing (no augmentation).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Returns transform pipeline for arbitrary input images before feeding to model.
    Resizes image to 32x32, converts to tensor, and applies CIFAR-10 normalization.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepares and returns training (augmented) and test (deterministic) DataLoaders.
    """
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    trainset = CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
    testset = CIFAR10(root=data_dir, train=False, download=download, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
