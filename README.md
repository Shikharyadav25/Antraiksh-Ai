# 🌌 AntraikshAI — Galaxy Morphology Classifier

AntraikshAI is a lightweight, presentable Streamlit application for demonstrating galaxy morphology classification. It accepts an uploaded astronomical image (or a bundled sample), runs a PyTorch ResNet-18 checkpoint when available, and displays a clean prediction UI with class probabilities and simple image-domain diagnostics.

> **Note:** This project is intended as a demo/portfolio app, not a production astronomy pipeline or scientific validation tool.

## ✨ Features

- **Streamlit dashboard** with a polished dark-space UI.
- **Galaxy image upload** for JPG, JPEG, PNG, and WebP files.
- **Sample image mode** using bundled repository images when available.
- **PyTorch inference** with a ResNet-18 model checkpoint.
- **Three morphology classes:** Elliptical, Irregular, and Spiral.
- **Simple diagnostics** for dark-background astronomical image suitability.
- **Graceful error handling** when a model checkpoint is missing or incompatible.

## 🧰 Tech Stack

| Layer | Tools |
| --- | --- |
| UI | Streamlit, custom CSS |
| ML / Inference | PyTorch, Torchvision ResNet-18 |
| Image Processing | Pillow, NumPy |
| Training / Evaluation Utilities | scikit-learn, Matplotlib, h5py, OpenCV |
| Language | Python 3.10+ recommended |

## 📁 Project Structure

```text
Antraiksh-Ai/
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── model/
│   ├── resnet_galaxy_classifier.pth # Preferred demo checkpoint
│   ├── galaxy_classifier.pth        # Alternate checkpoint
│   ├── train_resnet.py              # ResNet training script
│   ├── train.py                     # Custom CNN training script
│   ├── predict.py                   # CLI prediction helper
│   └── ...                          # Data conversion/evaluation scripts
├── scratch/                       # Verification/sample image experiments
├── generated/                     # Generated output placeholder
├── enhanced/                      # Enhanced output placeholder
└── README.md
```

## 🚀 Quick Start

### 1. Clone and enter the project

```bash
git clone <your-repository-url>
cd Antraiksh-Ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 🧪 How to Use the App

1. Start the app with `streamlit run app.py`.
2. Choose **Upload image** to provide your own galaxy image, or **Use sample** to try a bundled image.
3. Review the predicted morphology class and probability bars.
4. Open **Image diagnostics** to see simple dark-background checks.

For best visual results, use cropped galaxy images with a dark space background. Daylight scenes, people, landscapes, and other non-astronomical images may trigger a warning.

## 🤖 Model Checkpoints

The app searches for checkpoints in this order:

1. `model/resnet_galaxy_classifier.pth`
2. `resnet_galaxy_classifier.pth`
3. `model/galaxy_classifier.pth`
4. `galaxy_classifier.pth`
5. `best_model.pth` / `best_model.pt`
6. Any other `.pth` file in the repository

Expected checkpoint format:

```python
{
    "model_state_dict": model.state_dict(),
    "classes": ["elliptical", "irregular", "spiral"]
}
```

If no compatible checkpoint is found, the UI still opens and explains that inference is unavailable.

## 🏋️ Optional Training Workflow

These scripts are included for experimentation and are not required to run the Streamlit demo.

```bash
python model/convert_galaxy10.py
python model/prepare_data.py
python model/train_resnet.py
python model/confusion_matrix.py
```

You may need the Galaxy10/Galaxy10 DECals dataset and sufficient disk/GPU resources for training.

## ⚠️ Limitations

- The app uses a lightweight demo workflow and simple heuristics for image suitability.
- Predictions depend entirely on the bundled or supplied checkpoint quality.
- The domain check is not a scientific validation method.
- This project should not be used as a source of authoritative astronomical classification.

## 📌 Troubleshooting

- **`ModuleNotFoundError`**: Run `pip install -r requirements.txt` inside your activated environment.
- **App opens but inference is disabled**: Confirm a compatible `.pth` checkpoint exists in `model/`.
- **Slow startup**: PyTorch can take time to import on CPU-only machines.
- **Poor predictions**: Use clean, cropped galaxy images with dark backgrounds.

## 📄 License

Add your preferred license file before publishing or distributing this repository.
