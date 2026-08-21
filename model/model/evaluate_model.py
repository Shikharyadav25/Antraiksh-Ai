from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 1. Path Resolution
BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "model":
    BASE_DIR = BASE_DIR.parent

VAL_DIR = BASE_DIR / "data" / "val"
MODEL_PATH = BASE_DIR / "model" / "resnet_galaxy_classifier.pth"

# Hardware Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 2. Validation Preprocessing Pipeline
IMAGE_SIZE = 160
BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Path & File Checks
if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation folder not found at: {VAL_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model checkpoint not found at: {MODEL_PATH}")

dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
classes = dataset.classes

print(f"Classes ({len(classes)}):", classes)
print("Validation images:", len(dataset))

# 4. Initialize Model & Safely Load Checkpoint
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
except TypeError:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

# 5. Inference Loop
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# 6. Metrics & Report Generation
accuracy = (all_predictions == all_labels).mean() * 100

print("\n" + "=" * 45)
print("          Antraiksh AI Evaluation")
print("=" * 45)
print(f"Validation Accuracy: {accuracy:.2f}%\n")

print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))