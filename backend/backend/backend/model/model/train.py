import os

import torch
import torch.nn as nn

from torchvision import (
    datasets,
    transforms,
    models
)

from torch.utils.data import (
    DataLoader
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 32
EPOCHS = 10
IMAGE_SIZE = 224


train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(
        15
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


train_dataset = datasets.ImageFolder(
    "data/train",
    transform=train_transform
)


validation_dataset = datasets.ImageFolder(
    "data/val",
    transform=validation_transform
)


if len(train_dataset) == 0:

    raise RuntimeError(
        "Training dataset is empty."
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    "Classes:",
    train_dataset.classes
)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(validation_dataset)
)

print(
    "Device:",
    DEVICE
)


model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


for parameter in model.parameters():

    parameter.requires_grad = False


features = model.fc.in_features


model.fc = nn.Sequential(

    nn.Linear(
        features,
        128
    ),

    nn.ReLU(),

    nn.Dropout(
        0.3
    ),

    nn.Linear(
        128,
        len(
            train_dataset.classes
        )
    )
)


model = model.to(
    DEVICE
)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.0001
)


def evaluate():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    return (
        100 * correct / total
    )


best_accuracy = 0


for epoch in range(
    EPOCHS
):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

    accuracy = evaluate()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: "
        f"{running_loss / len(train_loader):.4f} "
        f"Validation: "
        f"{accuracy:.2f}%"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        os.makedirs(
            "model/weights",
            exist_ok=True
        )

        torch.save(
            {
                "model_state":
                    model.state_dict(),

                "classes":
                    train_dataset.classes
            },
            "model/weights/"
            "galaxy_classifier.pth"
        )


print(
    "Best validation accuracy:",
    best_accuracy
)