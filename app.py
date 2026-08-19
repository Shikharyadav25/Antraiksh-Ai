import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AntraikshAI • Galaxy Morphology Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CLASSES = ["elliptical", "irregular", "spiral"]
IMAGE_SIZE = 160

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FIND MODEL
# ============================================================

MODEL_CANDIDATES = [
    "resnet_galaxy_classifier.pth",
    "resnet_galaxy_classifier.pt",
    "model/resnet_galaxy_classifier.pth",
    "model/resnet_galaxy_classifier.pt",
    "best_model.pth",
    "best_model.pt",
    "model/best_model.pth",
    "galaxy_classifier.pth",
    "model/galaxy_classifier.pth",
]


def find_model_file():
    # First check explicit candidates
    for relative_path in MODEL_CANDIDATES:
        path = BASE_DIR / relative_path
        if path.exists() and path.is_file():
            return path

    # Next search recursively for .pth or .pt files
    possible_files = []
    for extension in ("*.pth", "*.pt"):
        possible_files.extend(BASE_DIR.rglob(extension))

    if not possible_files:
        return None

    # Prefer files with 'resnet' in their name
    resnet_files = [f for f in possible_files if "resnet" in f.name.lower()]
    if resnet_files:
        return resnet_files[0]

    keywords = ["best", "model", "galaxy", "classifier"]
    preferred = [
        f for f in possible_files
        if any(kw in f.name.lower() for kw in keywords)
    ]

    if preferred:
        return preferred[0]

    return possible_files[0]


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(num_classes=3):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )
    return model


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = find_model_file()

    if model_path is None:
        return None, None, DEFAULT_CLASSES, "No .pth or .pt model file was found in the workspace."

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False
        )

        classes = DEFAULT_CLASSES
        state_dict = None

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        else:
            if isinstance(checkpoint, dict):
                if "classes" in checkpoint and isinstance(checkpoint["classes"], list):
                    classes = checkpoint["classes"]

                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint

            if state_dict is None:
                return None, model_path, classes, "Could not extract state_dict from model file."

            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    key = key[7:]
                cleaned_state_dict[key] = value

            model = create_model(num_classes=len(classes))
            
            # Load state dict strictly if possible, fallback to non-strict
            try:
                model.load_state_dict(cleaned_state_dict, strict=True)
            except Exception:
                model.load_state_dict(cleaned_state_dict, strict=False)

        model = model.to(DEVICE)
        model.eval()

        return model, model_path, classes, None

    except Exception as error:
        return None, model_path, DEFAULT_CLASSES, f"Model loading error: {error}"


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CSS STYLING
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(68, 56, 160, 0.4), transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(14, 116, 144, 0.35), transparent 40%),
        radial-gradient(circle at 50% 80%, rgba(126, 34, 206, 0.25), transparent 50%),
        linear-gradient(135deg, #050716 0%, #090d26 50%, #041021 100%);
    color: #e2e8f0;
}

.block-container {
    max-width: 1100px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 36, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}

/* Input & radio text contrast */
[data-testid="stWidgetLabel"], [data-testid="stRadio"] label p, label p {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}

.logo-container {
    text-align: center;
    margin-top: 5px;
    margin-bottom: 20px;
}

.logo {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 60%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.logo-icon {
    font-size: 36px;
    margin-right: 10px;
    -webkit-text-fill-color: initial;
}

.subtitle {
    margin-top: 4px;
    color: #94a3b8;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.hero {
    text-align: center;
    padding: 35px 20px 25px 20px;
}

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.35);
    color: #c7d2fe;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 14px;
    color: #ffffff;
    line-height: 1.15;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 17px;
    max-width: 700px;
    margin: auto;
    line-height: 1.6;
}

.info-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.info-card:hover {
    border-color: rgba(129, 140, 248, 0.4);
    transform: translateY(-2px);
}

.model-number {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
}

.model-label {
    color: #94a3b8;
    font-size: 12px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-weight: 600;
}

.card-title {
    color: #818cf8;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-description {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.5;
}

.section-title {
    text-align: center;
    color: #ffffff;
    font-size: 26px;
    font-weight: 700;
    margin-top: 40px;
    margin-bottom: 8px;
}

.section-description {
    text-align: center;
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 24px;
}

.prediction-card {
    margin-top: 20px;
    padding: 26px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(30, 41, 59, 0.6));
    border: 1px solid rgba(129, 140, 248, 0.3);
    backdrop-filter: blur(10px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
    text-align: center;
}

.prediction-label {
    color: #a5b4fc;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
}

.prediction-name {
    color: #ffffff;
    font-size: 36px;
    font-weight: 800;
    margin: 8px 0;
}

.prediction-confidence {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 9999px;
    background: rgba(34, 197, 94, 0.2);
    border: 1px solid rgba(74, 222, 128, 0.4);
    color: #4ade80;
    font-size: 15px;
    font-weight: 700;
}

.probability-container {
    margin-top: 16px;
}

.probability-header {
    display: flex;
    justify-content: space-between;
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 6px;
}

.probability-bar {
    height: 10px;
    width: 100%;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    overflow: hidden;
}

.probability-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.fill-elliptical {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
}

.fill-irregular {
    background: linear-gradient(90deg, #ec4899, #f472b6);
}

.fill-spiral {
    background: linear-gradient(90deg, #06b6d4, #38bdf8);
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.025);
    border: 1px dashed rgba(129, 140, 248, 0.4);
    border-radius: 16px;
    padding: 15px;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(129, 140, 248, 0.7);
    background: rgba(255, 255, 255, 0.04);
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOGO & HERO
# ============================================================

st.markdown(
    """
<div class="logo-container">
    <div class="logo">
        <span class="logo-icon">🌌</span>AntraikshAI
    </div>
    <div class="subtitle">
        Deep Learning • Astronomy • Galaxy Intelligence
    </div>
</div>

<div class="hero">
    <div class="badge">
        ✦ DEEP LEARNING • ASTROPHYSICS • COMPUTER VISION
    </div>
    <div class="hero-title">
        Galaxy Classification
    </div>
    <div class="hero-subtitle">
        Decode the morphology of distant galaxies in real-time using
        Deep Residual Networks (ResNet18) and PyTorch.
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL & SIDEBAR
# ============================================================

model, model_path, CLASSES, model_error = load_model()

with st.sidebar:
    st.markdown("### 🌌 AntraikshAI Control Panel")
    st.markdown("---")
    st.markdown("#### 🔭 Hubble Morphology Guide")
    st.markdown("""
    - **Elliptical (E0-E7):** Smooth, featureless, elliptical light distributions with minimal gas/dust.
    - **Spiral (Sa-Sc / SBa-SBc):** Rotating disks with spiral arms, rich gas, and ongoing star formation.
    - **Irregular (Irr I/II):** Asymmetric shapes lacking a clear bulge or spiral structure.
    """)


# ============================================================
# MODEL CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
<div class="info-card">
    <div class="model-number">ResNet18</div>
    <div class="model-label">Deep Learning Model</div>
</div>
""",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
<div class="info-card">
    <div class="model-number">{len(CLASSES)}</div>
    <div class="model-label">Galaxy Classes</div>
</div>
""",
        unsafe_allow_html=True
    )

with col3:
    device_name = "GPU (CUDA)" if DEVICE.type == "cuda" else "CPU"
    st.markdown(
        f"""
<div class="info-card">
    <div class="model-number">{device_name}</div>
    <div class="model-label">Inference Device</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# MODEL STATUS
# ============================================================

if model_error:
    st.error(f"⚠️ {model_error}")
else:
    st.success(f"✅ Model weights active: **{model_path.name}**")


# ============================================================
# CLASSIFICATION SECTION
# ============================================================

st.markdown(
    """
<div class="section-title">
    Explore Galaxy Morphology
</div>

<div class="section-description">
    Upload an astronomical image or choose a sample galaxy from our dataset to perform inference.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# INPUT MODE SELECTION & SAMPLE IMAGES
# ============================================================

input_mode = st.radio(
    "Choose input source:",
    ["Upload Image", "Sample Galaxy Image"],
    horizontal=True,
    label_visibility="collapsed"
)

image_to_process = None
sample_caption = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose a galaxy image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            sample_caption = f"Uploaded File: {uploaded_file.name}"
        except Exception as error:
            st.error(f"Could not open uploaded image file: {error}")

else:
    # Look for sample test images
    sample_options = {}
    test_dir = BASE_DIR / "data" / "test"

    sample_presets = [
        ("Spiral Galaxy Sample", "spiral"),
        ("Elliptical Galaxy Sample", "elliptical"),
        ("Irregular Galaxy Sample", "irregular"),
    ]

    for label, category in sample_presets:
        cat_dir = test_dir / category
        if cat_dir.exists() and cat_dir.is_dir():
            files = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))
            if files:
                sample_options[f"✨ {label} ({category.capitalize()})"] = files[0]

    if sample_options:
        selected_sample_label = st.selectbox(
            "Select a sample image from the test set:",
            options=list(sample_options.keys())
        )
        sample_file_path = sample_options[selected_sample_label]
        try:
            image_to_process = Image.open(sample_file_path).convert("RGB")
            sample_caption = f"Sample: {sample_file_path.name}"
        except Exception as err:
            st.error(f"Error loading sample image: {err}")
    else:
        st.warning("No sample dataset images found in data/test directory.")


# ============================================================
# PREDICTION PROCESSOR
# ============================================================

if image_to_process is not None:

    image_col1, image_col2, image_col3 = st.columns([1, 2, 1])

    with image_col2:
        st.image(
            image_to_process,
            caption=sample_caption or "Selected Galaxy",
            use_container_width=True
        )

    if model is not None:
        try:
            input_tensor = transform(image_to_process).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_index = torch.max(probabilities, dim=1)

            pred_idx = predicted_index.item()
            conf_val = confidence.item() * 100
            predicted_class = CLASSES[pred_idx]

            icon_map = {
                "elliptical": "🪐",
                "irregular": "✨",
                "spiral": "🌀"
            }
            pred_icon = icon_map.get(predicted_class.lower(), "🌌")

            st.markdown(
                f"""
<div class="prediction-card">
    <div class="prediction-label">AntraikshAI Prediction</div>
    <div class="prediction-name">{pred_icon} {predicted_class.capitalize()}</div>
    <div class="prediction-confidence">Confidence: {conf_val:.2f}%</div>
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown(
                """
<div class="section-title" style="font-size:22px; margin-top:30px;">
    Classification Probabilities
</div>
""",
                unsafe_allow_html=True
            )

            prob_list = probabilities[0].detach().cpu().tolist()

            for class_name, prob in zip(CLASSES, prob_list):
                pct = prob * 100
                fill_class = f"fill-{class_name.lower()}"

                st.markdown(
                    f"""
<div class="probability-container">
    <div class="probability-header">
        <span>{class_name.capitalize()}</span>
        <span>{pct:.2f}%</span>
    </div>
    <div class="probability-bar">
        <div class="probability-fill {fill_class}" style="width:{pct:.2f}%"></div>
    </div>
</div>
""",
                    unsafe_allow_html=True
                )

        except Exception as error:
            st.error(f"Inference error: {error}")
    else:
        st.warning("Model is not loaded. Please verify model weights.")


# ============================================================
# GALAXY MORPHOLOGY CLASSES GUIDE
# ============================================================

st.markdown(
    """
<div class="section-title">
    Galaxy Morphologies
</div>

<div class="section-description">
    Understanding the structural properties classified by AntraikshAI.
</div>
""",
    unsafe_allow_html=True
)

class_col1, class_col2, class_col3 = st.columns(3)

with class_col1:
    st.markdown(
        """
<div class="info-card">
    <div class="card-title">🪐 Elliptical</div>
    <div class="card-description">
        Smooth, rounded light profiles ranging from spherical to elongated ellipsoids, containing older stellar populations.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

with class_col2:
    st.markdown(
        """
<div class="info-card">
    <div class="card-title">🌀 Spiral</div>
    <div class="card-description">
        Flat rotating disks containing bright spiral arms of gas, cosmic dust, and young blue star clusters.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

with class_col3:
    st.markdown(
        """
<div class="info-card">
    <div class="card-title">✨ Irregular</div>
    <div class="card-description">
        Disrupted or chaotic morphology without central symmetry, often shaped by gravitational tidal interactions.
    </div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    AntraikshAI • Deep Learning for Galaxy Morphology Classification
    <br><br>
    Built with PyTorch • Torchvision • Streamlit
</div>
""",
    unsafe_allow_html=True
)