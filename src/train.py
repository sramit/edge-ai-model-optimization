import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 10

DATA_DIR = "./data"
CHECKPOINT_DIR = "./results"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mobilenetv3_cifar10_fp32.pth")


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# -----------------------------
# Dataset
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])


train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform,
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=test_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


# -----------------------------
# Model
# -----------------------------
model = models.mobilenet_v3_small(weights=None)

# Adapt first convolution for CIFAR-10 resolution.
model.features[0][0] = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
)

# Remove ImageNet-style early downsampling.
model.features[0][1] = nn.BatchNorm2d(16)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES,
)

model = model.to(device)


# -----------------------------
# Training
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
)


def evaluate():
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return 100.0 * correct / total


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        accuracy = evaluate()

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {running_loss / len(train_loader):.4f} "
            f"Accuracy: {accuracy:.2f}% "
            f"Time: {epoch_time:.1f}s"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "accuracy": accuracy,
                    "epoch": epoch + 1,
                },
                CHECKPOINT_PATH,
            )

            print(f"Saved best model: {CHECKPOINT_PATH}")

    print(f"\nBest validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    train()
