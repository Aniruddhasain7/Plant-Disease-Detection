# 🌿 Plant Disease Detection System

An end-to-end Deep Learning & Computer Vision application designed to identify plant leaf diseases and provide immediate treatment and maintenance advisories. Built with **TensorFlow / Keras** and deployed using **Streamlit**.

---

## 🌟 Key Features

- 📸 **Dual Input Methods**: Upload high-resolution images (JPG, PNG, JPEG, WEBP) or capture directly using your device camera.
- 🔬 **Accurate Disease Diagnosis**: Deep Convolutional Neural Network (CNN) trained on the PlantVillage dataset.
- 📊 **Confidence & Probability Breakdown**: Real-time confidence scores and top-3 diagnostic predictions.
- 💊 **Treatment & Prevention Advisories**: Tailored recommendations for curing diseases and maintaining crop health.
- ⚡ **Lightweight Deployment**: Optimized model weights (~45.6 MB) for instant startup and seamless hosting on Streamlit Community Cloud.

---

## 🌿 Supported Crops & Diseases (15 Classes)

| Crop                 | Diagnostic Classes                                                                                                                                                           |
| :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🫑 Pepper (Bell)** | Bacterial Spot, Healthy                                                                                                                                                      |
| **🥔 Potato**        | Early Blight, Late Blight, Healthy                                                                                                                                           |
| **🍅 Tomato**        | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted Spider Mite), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 📂 Project Structure

```text
Plant-Disease-Detection/
├── .gitignore                                         # Git ignore configuration
├── app.py                                             # Streamlit Web Application
├── class_indices.json                                 # Class index-to-label mappings
├── Plant_Disease_Detection_using_Tensorflow.ipynb      # Training & evaluation Jupyter notebook
├── plant_disease_model.h5                             # Trained CNN model weights (Optimized ~45.6 MB)
├── requirements.txt                                   # Python dependencies for deployment
└── README.md                                          # Project documentation
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/Aniruddhasain7/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Create and Activate Virtual Environment

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

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Cloud

1. Fork or push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app** and select:
   - **Repository:** `YourUsername/Plant-Disease-Detection`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!** 🚀

---

## 🧠 Model Architecture & Training

- **Input Dimension:** $224 \times 224 \times 3$ (RGB)
- **Architecture:** Sequential Convolutional Neural Network
  - Conv2D (32 filters, $3\times3$) + ReLU $\rightarrow$ MaxPooling2D ($2\times2$)
  - Conv2D (64 filters, $3\times3$) + ReLU $\rightarrow$ MaxPooling2D ($2\times2$)
  - Flatten $\rightarrow$ Dense (64 units, ReLU) $\rightarrow$ Dense (15 units, Softmax)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Dataset:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
