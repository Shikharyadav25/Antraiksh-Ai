from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WEIGHTS = Path(
    "model/weights/galaxy_classifier.pth"
)


def load_model():

    if not WEIGHTS.exists():

        raise FileNotFoundError(
            "Trained model not found. "
            "Run model/prepare_data.py and "
            "model/train.py first."
        )

    checkpoint = torch.load(
        WEIGHTS,
        map_location=DEVICE
    )

    classes = checkpoint["classes"]

    model = models.resnet18(
        weights=None
    )

    features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(
            128,
            len(classes)
        )
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(DEVICE)

    model.eval()

    return model, classes