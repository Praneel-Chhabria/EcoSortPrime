import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoSort Pro | Smart Waste Analytics",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #1a1c20 0%, #0f1012 100%);
        color: #ffffff;
    }

    .nav-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .property-card {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 20px;
        padding: 0px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #333;
        overflow: hidden;
        transition: transform 0.3s ease;
        margin-top: 20px;
    }
    
    .card-header {
        background: #252525;
        padding: 20px;
        text-align: center;
        border-bottom: 1px solid #333;
    }

    .card-body {
        padding: 25px;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-top: 25px;
        padding-top: 20px;
        border-top: 1px solid #444;
    }
    .metric-item {
        text-align: center;
        width: 33%;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 16px;
        font-weight: bold;
        margin-top: 5px;
    }

    div.stButton > button:first-child {
        background-color: #00E676;
        color: #000;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        letter-spacing: 0.5px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #00C853;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION & MODEL LOADING ---
CATEGORIES_DETAIL = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
                     'metal', 'paper', 'plastic', 'shoes', 'trash']

CATEGORY_MAPPING = {
    'battery': 'Domestic Hazardous',
    'biological': 'Wet',
    'cardboard': 'Dry',
    'clothes': 'Dry',
    'glass': 'Dry',
    'metal': 'Dry',
    'paper': 'Dry',
    'plastic': 'Dry',
    'shoes': 'Dry',
    'trash': 'Dry'
}

@st.cache_resource
def load_tf_model():
    return tf.keras.models.load_model('waste_model_v1', custom_objects={'KerasLayer': hub.KerasLayer})

try:
    model = load_tf_model()
except Exception as e:
    st.error(f"⚠️ SYSTEM ERROR: Neural Network Model ('waste_model_v1') not found. Details: {e}")
    st.stop()

# --- HERO SECTION (Navbar) ---
st.markdown("""
<div class="nav-container">
    <h1 style="margin:0; font-family: 'Helvetica Neue', sans-serif; font-size: 2.5rem;">
        ♻️ EcoSort <span style="color:#00E676;">Prime</span>
    </h1>
    <p style="color: #888; margin-top:5px; font-size: 1.1rem;">AI-Powered Waste Segregation Dashboard</p>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 1. Data Source")
    st.write("Scan waste object for real-time analysis.")
    
    # Live camera input replaces the file uploader
    uploaded_file = st.camera_input("📸 Scan Item")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

with col2:
    st.markdown("### 2. Analytics Engine")
    
    if uploaded_file is None:
        st.info("System Standby. Awaiting Camera Input...")
    else:
        if st.button("🔍 Run Classification Model", type="primary", use_container_width=True):
            with st.spinner("Processing neural networks..."):
                try:
                    # --- PREPROCESSING ---
                    img_resized = image.convert('RGB').resize((224, 224))
                    img_array = np.array(img_resized)[np.newaxis, ...]
                    
                    # --- PREDICTION ---
                    predictions = model.predict(img_array, verbose=0)
                    detail_prediction_index = np.argmax(predictions[0])
                    detail_result = CATEGORIES_DETAIL[detail_prediction_index]
                    main_result = CATEGORY_MAPPING[detail_result]
                    
                    # --- DYNAMIC STYLING ---
                    if main_result == "Wet":
                        color = "#00E676" # Green
                        bin_type = "Green Bin (Compost)"
                        action = "Composting"
                    elif main_result == "Dry":
                        color = "#2979FF" # Blue
                        bin_type = "Blue Bin (Recyclable/Dry)"
                        action = "Sorting & Recovery"
                    else: # Domestic Hazardous
                        color = "#FF1744" # Red
                        bin_type = "Hazardous Bin"
                        action = "Safe Disposal"

                    # --- RENDER CARD ---
                    html_code = f"""
<div class="property-card">
    <div class="card-header">
        <h2 style="color: {color}; margin: 0; font-size: 2.2rem; font-weight: 700; text-transform: uppercase;">
            🗑️ {main_result}
        </h2>
    </div>
    <div class="card-body">
        <p style="text-align:center; color: #ccc; margin-top: 10px; font-size: 1rem; line-height: 1.6;">
            System has identified this item based on visual patterns. Following the recommended disposal protocol below ensures proper waste handling.
        </p>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-label">Target Bin</div>
                <div class="metric-value" style="color: {color};">{bin_type}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Action</div>
                <div class="metric-value" style="color: #fff;">{action}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Impact</div>
                <div class="metric-value" style="color: #fff;">Positive</div>
            </div>
        </div>
    </div>
</div>
"""
                    st.markdown(html_code, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
