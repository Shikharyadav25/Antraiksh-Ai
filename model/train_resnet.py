from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 1. Path Resolution & Directory Setup
BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "model" else Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "val"
MODEL_DIR = BASE_DIR / "model"

# Automatically create output model directory if missing
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 2. Hyperparameters & Hardware Device
IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# 3. Data Augmentation & Normalization
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Path Validation & Dataset Loading
if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder missing.\n"
        f"Expected:\n  Train: {TRAIN_DIR}\n  Val:   {VAL_DIR}\n"
        f"Run 'prepare_dataset.py' first to split the raw dataset."
    )

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Classes found ({len(train_dataset.classes)}):", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# 5. Model Initialization & Dynamic Output Layer
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Training & Validation Loop
best_val_accuracy = 0.0
save_path = MODEL_DIR / "resnet_galaxy_classifier.pth"

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Training Phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f}")

    train_accuracy = (100 * correct / total) if total > 0 else 0.0

    # Validation Phase
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = (100 * val_correct / val_total) if val_total > 0 else 0.0
    avg_train_loss = running_loss / total_batches if total_batches > 0 else 0.0

    print(
        f"Epoch {epoch + 1}/{EPOCHS} complete | "
        f"Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # Save Model Checkpoint
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "best_val_accuracy": best_val_accuracy
            },
            save_path
        )
        print(f"--> Saved new best model to: {save_path}")

print("\nTraining complete!")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"Model saved to: {save_path}")