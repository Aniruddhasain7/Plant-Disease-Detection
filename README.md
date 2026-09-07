# 🌿 Plant Disease Detection System

An end-to-end Deep Learning & Computer Vision application designed to identify plant leaf diseases in real-time and provide actionable treatment and maintenance advisories. Built using a **Custom Convolutional Neural Network (CNN)** in **TensorFlow / Keras** and deployed with a responsive, modern dark-themed web interface in **Streamlit**.

---

## 🌟 Key Features

- 📸 **Dual Input Modes**: Upload leaf images (`.jpg`, `.jpeg`, `.png`, `.webp`) or capture live photos using your device camera.
- 🧠 **Custom Deep CNN Backbone**: Multi-layer Convolutional Neural Network with dropout regularization trained for 5 epochs on the PlantVillage dataset.
- 🎯 **High Accuracy & Speed**: High diagnostic accuracy with sub-second inference latency.
- 📊 **Probability Distribution**: Real-time confidence score with a top-3 diagnostic prediction breakdown.
- 💊 **Actionable Agricultural Advisories**: Tailored treatment, prevention steps, and maintenance practices for every identified disease and healthy crop.
- ⚡ **Ultra-Lightweight (~10.3 MB)**: Model file is compact and optimized for rapid startup and low-memory environments like Streamlit Community Cloud.
- 🎨 **Modern Dark UI**: Styled with glassmorphic cards, custom badges, and smooth visual indicators for intuitive user experience.

---

## 🌿 Supported Crops & Diseases (15 Classes)

| Crop | Diagnostic Classes |
| :--- | :--- |
| **🫑 Pepper (Bell)** | Bacterial Spot, Healthy |
| **🥔 Potato** | Early Blight, Late Blight, Healthy |
| **🍅 Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted Spider Mite), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 📂 Project Structure

```text
Plant-Disease-Detection/
├── app.py
├── class_indices.json
├── Plant_Disease_Detection.ipynb
├── plant_disease_model.h5
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/Aniruddhasain7/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Open your web browser and navigate to:
```
http://localhost:8501
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. **Push your code to GitHub**: Make sure `app.py`, `plant_disease_model.h5`, `class_indices.json`, and `requirements.txt` are in your repository.
2. **Open Streamlit Community Cloud**: Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Create New App**:
   - **Repository:** `YourUsername/Plant-Disease-Detection`
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py`
4. Click **Deploy!** 🚀 The app will install the packages from `requirements.txt` and launch automatically.

---

## 🛠️ Tech Stack & Model Details

- **Language:** Python 3.10+ / 3.11
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Computer Vision & Image Processing:** Pillow (PIL), NumPy
- **Web App Framework:** Streamlit
- **Model Architecture:** Custom Deep CNN (4 Conv Blocks + Dense(256) + Dropout)
- **Training Setup:** 5 Epochs (Adam Optimizer, Categorical Crossentropy)
