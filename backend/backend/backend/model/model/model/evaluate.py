import torch

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from backend.model import (
    load_model,
    DEVICE
)


transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


dataset = datasets.ImageFolder(
    "data/test",
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


model, classes = load_model()


actual = []
predicted = []


with torch.no_grad():

    for images, labels in loader:

        outputs = model(
            images.to(DEVICE)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
            .cpu()
        )

        actual.extend(
            labels.tolist()
        )

        predicted.extend(
            predictions.tolist()
        )


print(
    classification_report(
        actual,
        predicted,
        target_names=classes,
        zero_division=0
    )
)


print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        actual,
        predicted
    )
)