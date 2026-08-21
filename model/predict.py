import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms, models

# 1. Configuration & Global Constants
IMAGE_SIZE = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["elliptical", "irregular", "spiral"]

# Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2. Initialize ResNet18 Architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

# 3. Path Resolution (Handles both execution modes & folder structures)
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

MODEL_PATH = BASE_DIR / "model" / "resnet_galaxy_classifier.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "resnet_galaxy_classifier.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# 4. Safe Checkpoint Loading
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


def validate_astronomical_image(image_pil):
    """
    Validates whether an input image exhibits deep-space astronomical features.
    Filters out non-astronomical uploads (daylight photos, sports, selfies).
    """
    img_gray = image_pil.convert("L")
    arr = np.array(img_gray, dtype=np.float32)
    h, w = arr.shape

    border_mask = np.ones_like(arr, dtype=bool)
    border_mask[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = False
    border_mean = float(np.mean(arr[border_mask]))

    dark_ratio = float(np.mean(arr < 65))
    is_astro = (dark_ratio >= 0.15) and (border_mean <= 110.0)
    return is_astro, dark_ratio, border_mean


def predict(image_input):
    """
    Infers galaxy type for file paths, byte streams, or pre-loaded PIL Image objects.
    Returns a structured dictionary compatible with Streamlit (app.py).
    """
    if isinstance(image_input, Image.Image):
        image_raw = image_input
    else:
        image_raw = Image.open(image_input)

    image_pil = ImageOps.exif_transpose(image_raw).convert("RGB")
    is_astro, dark_ratio, border_mean = validate_astronomical_image(image_pil)

    image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    galaxy = CLASSES[prediction.item()]
    confidence_score = float(confidence.item() * 100)

    # CLI Output Log
    print("\nAntraikshAI Prediction")
    print("----------------------")
    if not is_astro:
        print("Result:       Not a Galaxy")
        print("[WARNING] Non-Astronomical Image Detected.")
    else:
        print("Galaxy Type:", galaxy)
        print(f"Confidence:  {confidence_score:.2f}%")

    # Clean return dictionary for Streamlit/Web UI
    return {
        "is_astro": is_astro,
        "galaxy": galaxy if is_astro else "Not a Galaxy",
        "confidence": confidence_score if is_astro else 0.0,
        "raw_prediction": galaxy,
        "raw_confidence": confidence_score
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python model\\predict.py <image_path>")
    else:
        predict(sys.argv[1])