import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchvision import models
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TEST_DIR = PROJECT_ROOT / "data" / "test"
if not TEST_DIR.exists():
    TEST_DIR = SCRIPT_DIR / "data" / "test"

IMAGE_SIZE = 160
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    str(TEST_DIR),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load model weights
model_path = PROJECT_ROOT / "model" / "resnet_galaxy_classifier.pth"
if not model_path.exists():
    model_path = SCRIPT_DIR / "resnet_galaxy_classifier.pth"
if not model_path.exists():
    model_path = PROJECT_ROOT / "model" / "galaxy_classifier.pth"

checkpoint = torch.load(str(model_path), map_location=DEVICE)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

cleaned_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]
    cleaned_state_dict[k] = v

try:
    model.load_state_dict(cleaned_state_dict, strict=True)
except Exception:
    model.load_state_dict(cleaned_state_dict, strict=False)

model = model.to(DEVICE)
model.eval()

correct = 0
total = 0

class_correct = [0, 0, 0]
class_total = [0, 0, 0]

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        for i in range(labels.size(0)):

            label = labels[i].item()

            class_total[label] += 1

            if predicted[i] == labels[i]:
                class_correct[label] += 1


accuracy = 100 * correct / total

print()
print("========== TEST RESULTS ==========")
print("Test images:", total)
print(f"Test Accuracy: {accuracy:.2f}%")
print()

for i, name in enumerate(test_dataset.classes):

    acc = 100 * class_correct[i] / class_total[i]

    print(f"{name}: {acc:.2f}%")

print("==================================")