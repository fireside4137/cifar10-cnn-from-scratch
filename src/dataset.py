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


def get_transforms() -> transforms.Compose:
    """
    Returns standard normalization transform for CIFAR-10 images.
    Maps pixel values to [-1.0, 1.0] range.
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
    Prepares and returns training and test DataLoaders for CIFAR-10.
    """
    transform = get_transforms()

    trainset = CIFAR10(root=data_dir, train=True, download=download, transform=transform)
    testset = CIFAR10(root=data_dir, train=False, download=download, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
