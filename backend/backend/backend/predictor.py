import torch

from PIL import Image

from torchvision import transforms

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


model = None
classes = None


def predict(image_path):

    global model
    global classes

    if model is None:

        model, classes = load_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(
        image
    )

    image = image.unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        output = model(
            image
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        confidence, index = torch.max(
            probabilities,
            0
        )

    return {
        "prediction":
            classes[index.item()],

        "confidence":
            round(
                confidence.item() * 100,
                2
            ),

        "classes": [
            {
                "name": classes[i],
                "confidence":
                    round(
                        probabilities[i].item()
                        * 100,
                        2
                    )
            }
            for i in range(
                len(classes)
            )
        ]
    }