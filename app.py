from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

st.set_page_config(
    page_title="AntraikshAI | Galaxy Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 160
DEFAULT_CLASSES = ["elliptical", "irregular", "spiral"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CANDIDATES = [
    "model/resnet_galaxy_classifier.pth",
    "resnet_galaxy_classifier.pth",
    "model/galaxy_classifier.pth",
    "galaxy_classifier.pth",
    "best_model.pth",
    "best_model.pt",
]

CLASS_DESCRIPTIONS = {
    "elliptical": {
        "icon": "🪐",
        "title": "Elliptical",
        "text": "Smooth, rounded light profiles dominated by older stars and very little visible dust or arm structure.",
        "color": "#f59e0b",
    },
    "irregular": {
        "icon": "✨",
        "title": "Irregular",
        "text": "Asymmetric or disturbed galaxies without a clean disk, bulge, or spiral-arm pattern.",
        "color": "#ec4899",
    },
    "spiral": {
        "icon": "🌀",
        "title": "Spiral",
        "text": "Disk galaxies with a brighter central region and winding arms containing gas, dust, and young stars.",
        "color": "#06b6d4",
    },
}

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99, 102, 241, .30), transparent 32rem),
                radial-gradient(circle at top right, rgba(14, 165, 233, .18), transparent 28rem),
                linear-gradient(135deg, #050816 0%, #0f172a 55%, #020617 100%);
            color: #e5e7eb;
        }
        .block-container { max-width: 1180px; padding-top: 2rem; }
        [data-testid="stSidebar"] { background: rgba(2, 6, 23, .88); border-right: 1px solid rgba(148, 163, 184, .18); }
        [data-testid="stSidebar"] * { color: #dbeafe !important; }
        .hero {
            padding: 2rem; border: 1px solid rgba(148, 163, 184, .18); border-radius: 28px;
            background: linear-gradient(135deg, rgba(15, 23, 42, .84), rgba(30, 41, 59, .52));
            box-shadow: 0 28px 90px rgba(0,0,0,.35); margin-bottom: 1.5rem;
        }
        .eyebrow { color:#a5b4fc; font-weight:800; letter-spacing:.18em; font-size:.78rem; text-transform:uppercase; }
        .hero h1 { color:white; font-size: clamp(2.3rem, 6vw, 4.9rem); line-height:1; margin:.55rem 0; }
        .hero p { color:#cbd5e1; max-width: 760px; font-size:1.08rem; line-height:1.7; }
        .glass-card {
            height: 100%; padding: 1.15rem; border-radius: 20px; border: 1px solid rgba(148, 163, 184, .18);
            background: rgba(15, 23, 42, .68); box-shadow: 0 12px 34px rgba(0,0,0,.22);
        }
        .metric-title { color:#94a3b8; text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; font-weight:700; }
        .metric-value { color:#fff; font-size:1.55rem; font-weight:800; margin-top:.35rem; }
        .prediction {
            text-align:center; padding:1.6rem; border-radius:24px; border:1px solid rgba(129,140,248,.42);
            background:linear-gradient(135deg, rgba(79,70,229,.26), rgba(14,165,233,.12));
        }
        .prediction-name { color:#fff; font-size:2.25rem; font-weight:800; margin:.3rem 0; }
        .badge { display:inline-block; border-radius:999px; padding:.35rem .8rem; background:rgba(34,197,94,.16); color:#86efac; border:1px solid rgba(74,222,128,.34); font-weight:800; }
        .bar-bg { height: 12px; background: rgba(148,163,184,.16); border-radius:999px; overflow:hidden; margin:.35rem 0 1rem; }
        .bar-fill { height:100%; border-radius:999px; }
        .footer { color:#94a3b8; text-align:center; padding:2rem 0 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_model_file() -> Path | None:
    for candidate in MODEL_CANDIDATES:
        path = BASE_DIR / candidate
        if path.is_file():
            return path
    return next(BASE_DIR.glob("**/*.pth"), None)


def build_resnet(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource(show_spinner="Loading galaxy classifier...")
def load_model() -> tuple[nn.Module | None, Path | None, list[str], str | None]:
    model_path = find_model_file()
    if model_path is None:
        return None, None, DEFAULT_CLASSES, "No PyTorch checkpoint was found. The app will still open, but inference is disabled."

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        classes = checkpoint.get("classes", DEFAULT_CLASSES) if isinstance(checkpoint, dict) else DEFAULT_CLASSES
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else None
        if state_dict is None:
            return checkpoint.to(DEVICE).eval(), model_path, classes, None

        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model = build_resnet(len(classes))
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            st.toast("Loaded checkpoint with minor architecture differences.", icon="ℹ️")
        return model.to(DEVICE).eval(), model_path, classes, None
    except Exception as exc:
        return None, model_path, DEFAULT_CLASSES, f"Could not load checkpoint: {exc}"


def validate_astronomical_image(image: Image.Image) -> tuple[bool, float, float]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    border = np.ones_like(gray, dtype=bool)
    border[int(h * 0.2): int(h * 0.8), int(w * 0.2): int(w * 0.8)] = False
    dark_ratio = float(np.mean(gray < 70))
    border_mean = float(np.mean(gray[border]))
    return dark_ratio >= 0.15 and border_mean <= 120, dark_ratio, border_mean


def predict(image: Image.Image, model: nn.Module, classes: list[str]) -> tuple[str, float, list[float]]:
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    top_index = int(np.argmax(probabilities))
    return classes[top_index], float(probabilities[top_index]), probabilities.tolist()


def sample_images() -> dict[str, Path]:
    samples = {}
    for path in [BASE_DIR / "test_spiral.jpeg", BASE_DIR / "test_spiral.jpg", *sorted((BASE_DIR / "scratch" / "verification_samples").glob("*.jp*g"))]:
        if path.is_file():
            samples[path.name] = path
    return samples


def probability_bar(label: str, pct: float) -> None:
    key = label.lower()
    color = CLASS_DESCRIPTIONS.get(key, {}).get("color", "#818cf8")
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;font-weight:700;color:#e2e8f0;">
            <span>{label.capitalize()}</span><span>{pct:.1f}%</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
        """,
        unsafe_allow_html=True,
    )


inject_css()
model, model_path, classes, model_error = load_model()

with st.sidebar:
    st.title("🌌 AntraikshAI")
    st.caption("Simple Streamlit UI for galaxy morphology demos.")
    st.divider()
    st.write("**Model status**")
    if model_error:
        st.warning(model_error)
    else:
        st.success(f"Loaded `{model_path.name}`")
    st.write("**Device**", "CUDA GPU" if DEVICE.type == "cuda" else "CPU")
    st.write("**Classes**", ", ".join(c.capitalize() for c in classes))
    st.divider()
    st.info("Best results come from cropped, deep-space galaxy images with a dark background.")

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Computer Vision • Astronomy • Streamlit</div>
        <h1>Galaxy Morphology Classifier</h1>
        <p>Upload a galaxy image and get a quick, presentable morphology prediction across elliptical, irregular, and spiral classes. The backend stays intentionally lightweight for demos and portfolio reviews.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="glass-card"><div class="metric-title">Architecture</div><div class="metric-value">ResNet-18</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="glass-card"><div class="metric-title">Classes</div><div class="metric-value">{len(classes)}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="glass-card"><div class="metric-title">Runtime</div><div class="metric-value">Streamlit</div></div>', unsafe_allow_html=True)

st.subheader("Try the classifier")
input_mode = st.radio("Input source", ["Upload image", "Use sample"], horizontal=True)
image = None
caption = None

if input_mode == "Upload image":
    uploaded = st.file_uploader("Upload a JPG, PNG, or WebP galaxy image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        caption = uploaded.name
else:
    samples = sample_images()
    if samples:
        selected = st.selectbox("Choose a bundled sample", list(samples))
        image = ImageOps.exif_transpose(Image.open(samples[selected])).convert("RGB")
        caption = selected
    else:
        st.warning("No bundled sample images were found.")

if image:
    left, right = st.columns([0.95, 1.05], vertical_alignment="top")
    with left:
        st.image(image, caption=caption, use_container_width=True)
    with right:
        is_astro, dark_ratio, border_mean = validate_astronomical_image(image)
        if not is_astro:
            st.warning("This image does not look like a dark-background deep-space cutout, so treat any prediction as a demo only.")
        if model is None:
            st.error("Inference is unavailable because no compatible checkpoint was loaded.")
        else:
            label, confidence, probabilities = predict(image, model, classes)
            icon = CLASS_DESCRIPTIONS.get(label.lower(), {}).get("icon", "🌌")
            st.markdown(
                f'<div class="prediction"><div class="eyebrow">Prediction</div><div class="prediction-name">{icon} {label.capitalize()}</div><span class="badge">Confidence {confidence * 100:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            for class_name, probability in zip(classes, probabilities):
                probability_bar(class_name, probability * 100)
        with st.expander("Image diagnostics"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Dark background", f"{dark_ratio * 100:.1f}%")
            c2.metric("Border luminance", f"{border_mean:.1f}")
            c3.metric("Domain check", "Likely galaxy" if is_astro else "Review image")

st.subheader("Morphology quick guide")
g1, g2, g3 = st.columns(3)
for column, key in zip([g1, g2, g3], ["elliptical", "spiral", "irregular"]):
    item = CLASS_DESCRIPTIONS[key]
    column.markdown(
        f'<div class="glass-card"><div style="font-size:2rem">{item["icon"]}</div><h3 style="color:white;margin:.35rem 0">{item["title"]}</h3><p style="color:#cbd5e1">{item["text"]}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer">Built with Streamlit, PyTorch, Torchvision, Pillow, and NumPy.</div>', unsafe_allow_html=True)
