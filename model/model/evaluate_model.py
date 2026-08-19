import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

VAL_DIR = "data/val"
IMAGE_SIZE = 160
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

classes = dataset.classes

print("Classes:", classes)
print("Validation images:", len(dataset))

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

checkpoint = torch.load(
    "model/resnet_galaxy_classifier.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

accuracy = (
    np.array(all_predictions)
    == np.array(all_labels)
).mean() * 100

print("\nAccuracy:", f"{accuracy:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)