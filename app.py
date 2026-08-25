import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.01em;
    }

    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none;
    }

    .stApp {
        background-color: #000000;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 80% 80%, rgba(5, 150, 105, 0.05) 0%, transparent 35%);
        color: #ffffff;
    }

    .header-box {
        text-align: center;
        background: #080808;
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 25px rgba(0, 0, 0, 0.9), 0 0 15px rgba(16, 185, 129, 0.08);
    }

    .header-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.8);
        color: #34d399;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
    }

    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
    }
    
    .header-title span {
        color: #10b981;
    }

    .header-subtitle {
        color: #a1a1aa;
        font-size: 0.95rem;
        margin: 0 auto;
        max-width: 500px;
        line-height: 1.5;
    }

    .glass-card {
        background: #0a0a0a;
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.8);
    }

    .badge-healthy {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 5px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .badge-diseased {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 5px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .crop-pill {
        background: #141414;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #d4d4d8;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #34d399;
    }

    .advisory-item {
        background: #080808;
        border-left: 3px solid #10b981;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        padding: 0.75rem 0.9rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.4rem;
        color: #d4d4d8;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .stButton>button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        border: 1px solid #10b981 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }

    [data-testid="stFileUploader"] section {
        background: #080808 !important;
        border: 2px dashed rgba(16, 185, 129, 0.3) !important;
        border-radius: 14px !important;
        padding: 1.25rem !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #10b981 !important;
    }

    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }

    [data-testid="stImage"] img {
        max-width: 260px !important;
        max-height: 260px !important;
        object-fit: cover !important;
        border-radius: 14px !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7) !important;
    }

    .prob-bar-container {
        margin-bottom: 0.65rem;
    }
    
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
        color: #e4e4e7;
    }
    
    .prob-bar-bg {
        width: 100%;
        height: 7px;
        background: #141414;
        border-radius: 9999px;
        overflow: hidden;
    }
    
    .prob-bar-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #059669 0%, #10b981 50%, #34d399 100%);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: #080808;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        color: #a1a1aa;
        padding: 6px 14px;
        font-size: 0.9rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.12) !important;
        border-color: #10b981 !important;
        color: #34d399 !important;
    }
</style>
""", unsafe_allow_html=True)

DISEASE_DETAILS = {
    "Pepper__bell___Bacterial_spot": {
        "title": "Bell Pepper — Bacterial Spot",
        "crop": "Bell Pepper",
        "status": "Diseased",
        "symptoms": "Water-soaked lesions on leaves turning dark brown with yellow halos.",
        "tips": [
            "Apply copper-based bactericides early in the disease cycle.",
            "Remove and safely destroy infected plant foliage.",
            "Avoid overhead watering to keep leaf surfaces dry.",
            "Rotate crops for at least 2 years with non-solanaceous crops."
        ]
    },
    "Pepper__bell___healthy": {
        "title": "Bell Pepper — Healthy",
        "crop": "Bell Pepper",
        "status": "Healthy",
        "symptoms": "Vibrant, uniform green foliage with no spotting or discoloration.",
        "tips": [
            "Maintain regular watering directly at the base of the plant.",
            "Ensure 6-8 hours of direct daily sunlight.",
            "Apply balanced fertilizer during flowering and fruit set."
        ]
    },
    "Potato___Early_blight": {
        "title": "Potato — Early Blight",
        "crop": "Potato",
        "status": "Diseased",
        "symptoms": "Target-board concentric dark brown rings on mature leaves.",
        "tips": [
            "Apply protectant fungicides like Mancozeb or copper sprays.",
            "Prune infected lower leaves to increase airflow.",
            "Mulch around plants to prevent soil splashing onto foliage.",
            "Maintain crop rotation."
        ]
    },
    "Potato___Late_blight": {
        "title": "Potato — Late Blight",
        "crop": "Potato",
        "status": "Diseased",
        "symptoms": "Large, dark water-soaked patches on leaves with white fungal growth underneath.",
        "tips": [
            "Apply systemic fungicides (e.g., Metalaxyl or Dimethomorph).",
            "Promptly destroy severely affected plants to stop spreading.",
            "Avoid overhead irrigation and ensure good soil drainage.",
            "Use certified disease-free seed tubers."
        ]
    },
    "Potato___healthy": {
        "title": "Potato — Healthy",
        "crop": "Potato",
        "status": "Healthy",
        "symptoms": "Robust green foliage with clean, vigorous leaf growth.",
        "tips": [
            "Hill soil around growing stems regularly.",
            "Water deeply once or twice a week without waterlogging.",
            "Regularly monitor foliage for early pest symptoms."
        ]
    },
    "Tomato_Bacterial_spot": {
        "title": "Tomato — Bacterial Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Small, greasy dark brown spots with yellow halos on foliage.",
        "tips": [
            "Use copper sprays combined with Mancozeb for protection.",
            "Sanitize pruning shears between each plant.",
            "Use drip irrigation rather than overhead sprinklers.",
            "Clean up and remove crop debris after harvest."
        ]
    },
    "Tomato_Early_blight": {
        "title": "Tomato — Early Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Concentric bullseye rings on lower leaves, surrounded by yellow halos.",
        "tips": [
            "Spray with bio-fungicides or Chlorothalonil / Azoxystrobin.",
            "Stake and prune lower leaves to promote good airflow.",
            "Add mulch to prevent fungal spores in soil from reaching leaves.",
            "Water at the base early in the morning."
        ]
    },
    "Tomato_Late_blight": {
        "title": "Tomato — Late Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Irregular dark greenish-black water-soaked lesions that rot rapidly.",
        "tips": [
            "Apply copper or systemic fungicides immediately upon detection.",
            "Remove infected foliage promptly on dry days.",
            "Space plants adequately to encourage sunlight and wind flow.",
            "Do not compost infected plant parts."
        ]
    },
    "Tomato_Leaf_Mold": {
        "title": "Tomato — Leaf Mold",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Pale yellow patches on leaf tops with velvety olive mold underneath.",
        "tips": [
            "Improve greenhouse ventilation and lower humidity levels.",
            "Apply copper-based fungicides or sulfur dusts.",
            "Space plants well to promote rapid leaf drying.",
            "Sanitize garden tools and stakes regularly."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "title": "Tomato — Septoria Leaf Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Tiny circular spots with dark brown margins and grey-white centers.",
        "tips": [
            "Remove infected lower leaves to limit upward spread.",
            "Apply organic copper fungicide or Chlorothalonil every 7-10 days.",
            "Mulch soil around plants to prevent splashing.",
            "Practice crop rotation with non-solanaceous crops."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "title": "Tomato — Two-Spotted Spider Mite",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Fine yellow stippling on leaf surface with delicate webbing underneath.",
        "tips": [
            "Spray with Neem oil, insecticidal soap, or horticultural oils.",
            "Spray water on leaf undersides to dislodge spider mites.",
            "Keep plants well-hydrated during hot, dry weather.",
            "Introduce beneficial predatory mites if available."
        ]
    },
    "Tomato__Target_Spot": {
        "title": "Tomato — Target Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Circular brown lesions with concentric zonation on foliage and fruit.",
        "tips": [
            "Apply recommended fungicides (Chlorothalonil or Mancozeb).",
            "Prune lower leaves to improve airflow around the base.",
            "Avoid overhead watering and avoid excess nitrogen fertilizer."
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "title": "Tomato — Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Severe upward leaf curling, yellowed margins, and stunted growth.",
        "tips": [
            "Control whitefly vectors using yellow sticky traps and neem oil.",
            "Remove and safely dispose of infected plants.",
            "Use fine insect netting in nursery beds.",
            "Choose virus-resistant tomato varieties for future planting."
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "title": "Tomato — Mosaic Virus",
        "crop": "Tomato",
        "status": "Diseased",
        "symptoms": "Mottled dark and light green mosaic patterns with distorted leaves.",
        "tips": [
            "Remove and destroy infected plants (no chemical cure available).",
            "Disinfect hands and tools with a 10% bleach solution.",
            "Avoid smoking or handling tobacco near tomato plants.",
            "Always use certified virus-free seeds."
        ]
    },
    "Tomato_healthy": {
        "title": "Tomato — Healthy",
        "crop": "Tomato",
        "status": "Healthy",
        "symptoms": "Clean, vigorous dark green leaves with strong plant structure.",
        "tips": [
            "Water consistently at soil level (1-2 inches per week).",
            "Provide sturdy staking or caging support.",
            "Feed with balanced potassium/calcium fertilizer for healthy growth."
        ]
    }
}

MODEL_PATH = "plant_disease_model.h5"
CLASSES_PATH = "class_indices.json"

@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_indices():
    if not os.path.exists(CLASSES_PATH):
        return {}
    with open(CLASSES_PATH, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, class_indices, image: Image.Image):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img, verbose=0)[0]
    
    top_idx = int(np.argmax(predictions))
    top_class_key = class_indices.get(top_idx, f"Class {top_idx}")
    top_confidence = float(predictions[top_idx])
    
    top_3_indices = np.argsort(predictions)[::-1][:3]
    top_3 = []
    for idx in top_3_indices:
        c_name = class_indices.get(int(idx), f"Class {idx}")
        info = DISEASE_DETAILS.get(c_name, {})
        title = info.get("title", c_name.replace("_", " "))
        top_3.append({
            "name": title,
            "confidence": float(predictions[idx]) * 100
        })
        
    return top_class_key, top_confidence, top_3

model = load_prediction_model()
class_indices = load_class_indices()

st.markdown("""
<div class="header-box">
    <div class="header-badge">AI Diagnostic System</div>
    <h1 class="header-title">Plant Disease <span>Detection</span></h1>
    <p class="header-subtitle">Upload or capture a leaf photo for instant deep learning analysis and treatment guidance.</p>
</div>
""", unsafe_allow_html=True)

if model is None or not class_indices:
    st.error("⚠️ Model file (`plant_disease_model.h5`) or `class_indices.json` not found in workspace.")
    st.stop()

input_tab1, input_tab2 = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

uploaded_file = None
with input_tab1:
    uploaded_file = st.file_uploader("Upload leaf photo (JPG, PNG, JPEG, WEBP)", type=["jpg", "jpeg", "png", "webp"], key="uploader")

with input_tab2:
    camera_file = st.camera_input("Capture leaf photo", key="camera")
    if camera_file is not None:
        uploaded_file = camera_file

if uploaded_file is not None:
    active_image = Image.open(uploaded_file)
    st.image(active_image, caption="Input Leaf", width=260)
    
    with st.spinner("Analyzing with Neural Network..."):
        raw_class, confidence, top_3 = predict(model, class_indices, active_image)
        details = DISEASE_DETAILS.get(raw_class, {
            "title": raw_class.replace("_", " "),
            "crop": "Plant",
            "status": "Healthy" if "healthy" in raw_class.lower() else "Diseased",
            "symptoms": "No visible symptoms.",
            "tips": ["Monitor plant health regularly."]
        })
        
        conf_percent = confidence * 100
        is_healthy = details["status"] == "Healthy"
        status_html = f'<div class="badge-healthy">🟢 {details["status"]}</div>' if is_healthy else f'<div class="badge-diseased">🔴 {details["status"]}</div>'
        
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 0.8rem;">
                {status_html}
            </div>
            <h2 style="font-size: 1.6rem; margin: 0 0 0.7rem 0; color: {'#34d399' if is_healthy else '#f87171'};">
                {details['title']}
            </h2>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem;">
                <span style="color: #a1a1aa; font-size: 0.85rem;">Confidence Score</span>
                <span class="metric-value">{conf_percent:.1f}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: {min(conf_percent, 100):.1f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**{'🌱 Maintenance Advisory:' if is_healthy else '💊 Treatment & Prevention:'}**")
        for tip in details["tips"]:
            st.markdown(f'<div class="advisory-item">✓ {tip}</div>', unsafe_allow_html=True)
            
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📊 Probability Breakdown")
        for item in top_3:
            st.markdown(f"""
            <div class="prob-bar-container">
                <div class="prob-label">
                    <span>{item['name']}</span>
                    <span style="font-weight: 700; color: #34d399;">{item['confidence']:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {item['confidence']:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 2.5rem 1.5rem;">
        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🌿</div>
        <h3 style="color: #ffffff; margin-bottom: 0.4rem;">No Leaf Selected</h3>
        <p style="color: #a1a1aa; font-size: 0.9rem; margin: 0 auto; max-width: 320px;">
            Upload or capture a leaf photo above to view diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)
