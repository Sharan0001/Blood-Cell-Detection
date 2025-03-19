import streamlit as st
import subprocess
import os
import pandas as pd
from PIL import Image
import base64

# Configure Page
st.set_page_config(
    page_title="Blood Cell Detection",
    page_icon="🩸",
    layout="wide"
)

# Function to set background image
def set_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    bg_image_style = f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Set background image (Hugging Face compatible path)
set_bg_image("UI.jpg")

# Model path (Hugging Face compatible path)
MODEL_PATH = "best.pt"
OUTPUT_DIR = "runs/detect/predict"

# Function to run YOLO inference using CLI
def run_yolo_cli(image_path):
    if os.path.exists(OUTPUT_DIR):
        subprocess.run(["rm", "-rf", OUTPUT_DIR])  # Remove previous results

    cmd = f"yolo predict model={MODEL_PATH} source={image_path} save=True"
    subprocess.run(cmd, shell=True)

    output_files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
    detected_image = [f for f in output_files if f.endswith(".jpg")]

    if detected_image:
        return os.path.join(OUTPUT_DIR, detected_image[0])
    return None

# Function to calculate precision & recall
def calculate_metrics():
    data = {
        "Class": ["Platelets", "RBC", "WBC"],
        "Precision": [0.831, 0.726, 0.961],
        "Recall": [0.892, 0.858, 0.990],
        "mAP50": [0.917, 0.852, 0.992],
        "mAP50-95": [0.517, 0.628, 0.822]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate overall precision & recall (mean values)
    overall_precision = df["Precision"].mean()
    overall_recall = df["Recall"].mean()
    overall_map50 = df["mAP50"].mean()
    overall_map50_95 = df["mAP50-95"].mean()
    
    # Append "Combined" row
    combined_data = {
        "Class": ["Overall"],
        "Precision": [overall_precision],
        "Recall": [overall_recall],
        "mAP50": [overall_map50],
        "mAP50-95": [overall_map50_95]
    }
    
    df_combined = pd.DataFrame(combined_data)
    
    # Concatenate original and overall metrics
    final_df = pd.concat([df, df_combined], ignore_index=True)
    
    return final_df

# Sidebar
st.sidebar.title("⚙️ Settings")

uploaded_image = st.sidebar.file_uploader(
    "📂 Upload an Image", type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    st.sidebar.success("✅ Image uploaded successfully!")

# Main Title
st.title("🔬 Blood Cell Detection")

# Run Detection
if uploaded_image:
    image_path = "temp.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.read())

    # Two-column layout for images
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_path, use_container_width=True)
        st.markdown(
            "<div style='background-color: #008080; color: white; text-align: center; padding: 8px; border-radius: 8px; font-weight: bold; margin-bottom: 20px;'>📷 Uploaded Image</div>",
            unsafe_allow_html=True
        )

    if st.button("🚀 Detect Blood Cells", help="Click to start detection"):
        with st.spinner("🧐 Analyzing Image... Please wait."):
            detected_image_path = run_yolo_cli(image_path)

        if detected_image_path:
            with col2:
                st.image(Image.open(detected_image_path), use_container_width=True)
                st.markdown(
                    "<div style='background-color: #004080; color: white; text-align: center; padding: 8px; border-radius: 8px; font-weight: bold;'>🔍 Detected Objects</div>",
                    unsafe_allow_html=True
                )

            # Display precision-recall table with enhanced styling
            st.subheader("📊 Model Performance Metrics")

            styled_table = calculate_metrics().style.set_table_styles(
                [
                    {"selector": "th", "props": [("background-color", "rgba(255, 255, 255, 0.9)"), 
                                                 ("color", "black"), 
                                                 ("font-weight", "bold"), 
                                                 ("border", "1px solid black"), 
                                                 ("padding", "10px"), 
                                                 ("text-align", "center") ]},
                    {"selector": "td", "props": [("background-color", "rgba(255, 255, 255, 0.9)"), 
                                                 ("color", "black"), 
                                                 ("border", "1px solid black"), 
                                                 ("padding", "10px"), 
                                                 ("text-align", "center") ]} 
                ],
                overwrite=True  # Ensures centering applies correctly
            )

            st.table(styled_table)

        else:
            st.error("❌ Error: No detections found or model did not generate output.")