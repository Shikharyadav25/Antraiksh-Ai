import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

IMAGE_SIZE = 160

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "elliptical",
    "irregular",
    "spiral"
]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

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


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    galaxy = CLASSES[prediction.item()]

    confidence = confidence.item() * 100

    print("\nAntraikshAI Prediction")
    print("----------------------")
    print("Galaxy Type:", galaxy)
    print("Confidence:", f"{confidence:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print("python model\\predict.py <image_path>")

    else:

        predict(sys.argv[1])