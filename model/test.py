import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TEST_DIR = "data/test"
IMAGE_SIZE = 64
BATCH_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class GalaxyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = GalaxyCNN().to(DEVICE)

checkpoint = torch.load(
    "model/galaxy_classifier.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

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