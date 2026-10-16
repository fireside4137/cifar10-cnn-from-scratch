import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .model import CNN
    from .dataset import get_data_loaders
except ImportError:
    from model import CNN
    from dataset import get_data_loaders


def get_device() -> torch.device:
    """Detect and return the best available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    data_dir: str = "./data",
    weights_path: str = "weights/cifar10_cnn.pth"
):
    device = get_device()
    print(f"--> Using compute device: {device}")

    # Prepare loaders
    trainloader, testloader = get_data_loaders(data_dir=data_dir, batch_size=batch_size)

    # Initialize model, criterion, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(f"--> Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        running_train_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(trainloader)
        train_losses.append(epoch_train_loss)

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(testloader)
        val_losses.append(epoch_val_loss)
        accuracy = (correct / total) * 100

        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Accuracy: {accuracy:.2f}%"
        )

        # Save best model checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(model.state_dict(), weights_path)

    print(f"\n--> Training complete! Best weights saved to: {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data folder")
    parser.add_argument("--output", type=str, default="weights/cifar10_cnn.pth", help="Checkpoint output path")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        weights_path=args.output
    )
