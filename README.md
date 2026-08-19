# 🌌 AntraikshAI • Galaxy Morphology Classifier

AntraikshAI is an AI-powered astronomical system designed for classifying galaxy morphologies (Elliptical, Irregular, and Spiral) using deep convolutional neural networks (ResNet-18) implemented in PyTorch, accompanied by an interactive Streamlit web dashboard.

---

## 🚀 Features

- **Galaxy Classification**: Classifies celestial images into Elliptical, Irregular, or Spiral galaxies.
- **Deep Neural Networks**: Powered by custom CNN and fine-tuned ResNet-18 architectures in PyTorch.
- **Interactive Web Interface**: Streamlit application (`app.py`) for uploading images, viewing predictions, confidence scores, and feature visualization.
- **Dataset Pipeline**: Scripts to extract from `Galaxy10_DECals` HDF5 dataset, preprocess images, generate splits, and compute evaluation metrics.

---

## 🛠️ Project Structure

```text
AntraikshAI/
├── app.py                      # Main Streamlit web application
├── model/                      # Model training, evaluation, and weights
│   ├── resnet_galaxy_classifier.pth  # Pre-trained ResNet model checkpoint
│   ├── galaxy_classifier.pth         # Custom CNN model checkpoint
│   ├── train_resnet.py         # ResNet model training script
│   ├── train.py                # Custom CNN training script
│   ├── predict.py              # Single image inference script
│   ├── prepare_data.py         # Dataset train/val/test splitter
│   ├── convert_galaxy10.py     # HDF5 Galaxy10 extraction script
│   └── confusion_matrix.py     # Confusion matrix evaluation generator
├── backend/                    # Backend modular services and image enhancer
├── frontend/                   # Frontend assets and UI components
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📦 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AntraikshAI.git
   cd AntraikshAI
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎈 Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## 📊 Dataset & Model Training

1. **Data Conversion**: Extract images from `Galaxy10_DECals.h5`:
   ```bash
   python model/convert_galaxy10.py
   ```

2. **Data Preparation**: Split dataset into train, val, and test subsets:
   ```bash
   python model/prepare_data.py
   ```

3. **Train ResNet Classifier**:
   ```bash
   python model/train_resnet.py
   ```

4. **Evaluate Model Performance**:
   ```bash
   python model/confusion_matrix.py
   ```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
