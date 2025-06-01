import os
import sys
import streamlit as st
from google.cloud import vision
import pickle
from PIL import Image
from io import BytesIO
from pathlib import Path

# --- Improved Path Handling ---
def find_file(filename, search_paths):
    """Search for a file in multiple locations"""
    for path in search_paths:
        if path.exists():
            return path
    return None

# --- Google Cloud Credentials Setup ---
def setup_gcp_credentials():
    """Locate and set up GCP credentials"""
    credential_file = "food-glucose-predictor-30b3feae0ca8.json"
    
    # Check possible locations
    possible_paths = [
        Path(credential_file),  # Current directory
        Path("src") / credential_file,  # src subdirectory
        Path("../") / credential_file,  # Parent directory
        Path("../src") / credential_file  # Parent's src directory
    ]
    
    cred_path = find_file(credential_file, possible_paths)
    if cred_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
        return True
    
    st.error(f"""
    Google Cloud credentials file not found. Please ensure:
    1. The file '{credential_file}' exists in one of these locations:
       - {Path.cwd()}
       - {Path.cwd() / 'src'}
       - {Path.cwd().parent}
       - {Path.cwd().parent / 'src'}
    2. The file has the correct permissions
    """)
    return False

if not setup_gcp_credentials():
    st.stop()

# --- Model Loading ---
def load_model_files():
    """Load model files with proper path resolution"""
    model_name = "food_glucose_model.pkl"
    vectorizer_name = "food_vectorizer.pkl"
    
    # Try multiple possible locations
    possible_paths = [
        Path(model_name),  # Same directory
        Path("models") / model_name,  # models subdirectory
        Path("../") / model_name,  # Parent directory
        Path("../models") / model_name  # Parent's models directory
    ]
    
    model_path = find_file(model_name, possible_paths)
    if not model_path:
        raise FileNotFoundError(f"Could not find {model_name} in any standard location")
    
    # Find vectorizer in same directory as model
    vectorizer_path = model_path.parent / vectorizer_name
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Could not find {vectorizer_name} in {vectorizer_path.parent}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

try:
    model, vectorizer = load_model_files()
except FileNotFoundError as e:
    st.error(f"""
    Model files not found: {e}
    
    Please ensure both files exist in one of these locations:
    - Current directory: {Path.cwd()}
    - models/ subdirectory: {Path.cwd() / 'models'}
    - Parent directory: {Path.cwd().parent}
    - Parent's models/ directory: {Path.cwd().parent / 'models'}
    """)
    st.stop()

# --- Food Detection Function ---
def detect_food(image_bytes):
    """Detect food using Google Cloud Vision"""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    
    try:
        # First try web detection (better for prepared dishes)
        web_response = client.web_detection(image=image)
        if web_response.web_detection.web_entities:
            for entity in web_response.web_detection.web_entities:
                if entity.score > 0.75:
                    return entity.description.capitalize()
        
        # Then try label detection
        label_response = client.label_detection(image=image)
        food_labels = [
            label.description.capitalize()
            for label in label_response.label_annotations
            if label.score > 0.85 and any(
                kw in label.description.lower() 
                for kw in ['food', 'dish', 'cuisine', 'meal']
            )
        ]
        
        return food_labels[0] if food_labels else "Unknown Food"
    
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return "Detection Failed"

# --- Recommendation System ---
def get_diabetic_recommendation(glucose, food_name):
    """Generate diabetic-friendly recommendations"""
    glycemic_load = (glucose * 50) / 100
    
    if "injera" in food_name.lower():
        return {
            "recommendation": "✅ Recommended",
            "details": "Teff injera has low glycemic index (GI 50-57). Good for diabetics in moderation.",
            "glycemic_load": round(glycemic_load, 1)
        }
    elif glucose < 10:
        return {
            "recommendation": "✅ Recommended",
            "details": f"Very low glucose content ({glucose}g/100g). Safe for diabetics.",
            "glycemic_load": round(glycemic_load, 1)
        }
    elif 10 <= glucose <= 20:
        return {
            "recommendation": "⚠️ Moderate",
            "details": f"Moderate glucose content ({glucose}g/100g). Consume in controlled portions.",
            "glycemic_load": round(glycemic_load, 1)
        }
    else:
        return {
            "recommendation": "❌ Not Recommended",
            "details": f"High glucose content ({glucose}g/100g). May cause blood sugar spikes.",
            "glycemic_load": round(glycemic_load, 1)
        }

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Food Glucose Predictor",
        page_icon="🍏",
        layout="wide"
    )
    
    st.title("🍏 Food Glucose Predictor")
    st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    .result-box { 
        padding: 20px; 
        border-radius: 10px; 
        margin: 10px 0px; 
        background-color: #f8f9fa;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        border-left: 5px solid #FFC107;
    }
    .danger-box {
        border-left: 5px solid #F44336;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Image Input Section
    with st.expander("📷 Analyze Food Image", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader(
                "Upload food photo", 
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of a single food item"
            )
        with col2:
            camera_img = st.camera_input(
                "Or take a photo",
                help="Position food in center of frame"
            )
        
        if uploaded_file or camera_img:
            image_bytes = uploaded_file.read() if uploaded_file else camera_img.getvalue()
            
            with st.spinner("🔍 Analyzing food..."):
                try:
                    img = Image.open(BytesIO(image_bytes))
                    st.image(img, caption="Your food", width=300, use_column_width=True)
                    
                    food_name = detect_food(image_bytes)
                    if food_name not in ["Unknown Food", "Detection Failed"]:
                        st.success(f"Detected: **{food_name}**")
                        
                        glucose = model.predict(vectorizer.transform([food_name.lower()]))[0]
                        recommendation = get_diabetic_recommendation(glucose, food_name)
                        
                        box_class = ("warning-box" if "⚠️" in recommendation['recommendation'] else 
                                    "danger-box" if "❌" in recommendation['recommendation'] else "")
                        
                        st.markdown(f"""
                        <div class="result-box {box_class}">
                            <h3>Nutrition Analysis</h3>
                            <p class="big-font">Glucose: <strong>{glucose:.1f}g/100g</strong></p>
                            <p class="big-font">Glycemic Load: <strong>{recommendation['glycemic_load']}</strong></p>
                            <p class="big-font">Verdict: <strong style="color:{{
                                'green' if '✅' in recommendation['recommendation'] 
                                else 'orange' if '⚠️' in recommendation['recommendation'] 
                                else 'red'
                            }}">{recommendation['recommendation']}</strong></p>
                            <p>{recommendation['details']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Couldn't identify food. Try manual input below.")
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

    # Manual Input Section
    with st.expander("✍️ Manual Input", expanded=False):
        food_text = st.text_input("Enter food name:", placeholder="e.g., Injera, Pasta, Apple")
        if st.button("Predict") and food_text:
            with st.spinner("Calculating..."):
                try:
                    glucose = model.predict(vectorizer.transform([food_text.lower()]))[0]
                    recommendation = get_diabetic_recommendation(glucose, food_text)
                    
                    box_class = ("warning-box" if "⚠️" in recommendation['recommendation'] else 
                                "danger-box" if "❌" in recommendation['recommendation'] else "")
                    
                    st.markdown(f"""
                    <div class="result-box {box_class}">
                        <h3>Results for {food_text.capitalize()}</h3>
                        <p class="big-font">Glucose: <strong>{glucose:.1f}g/100g</strong></p>
                        <p class="big-font">Recommendation: <strong>{recommendation['recommendation']}</strong></p>
                        <p>{recommendation['details']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()