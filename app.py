import streamlit as st
import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🌹 Rose Counter",
    layout="wide",
    page_icon="🌹"
)

st.markdown("<h1 style='text-align: center;'>🌹 Advanced Rose Detection & Counting</h1>", unsafe_allow_html=True)
st.markdown("---")


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("Yield_model.pt")

    detection_model = UltralyticsDetectionModel(
        model_path="Yield_model.pt",
        confidence_threshold=0.45,
        device="cpu"
    )

    return model, detection_model


model, detection_model = load_model()


# -----------------------------
# Upload Section
# -----------------------------
st.sidebar.header("📂 Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose a rose image",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    # -----------------------------
    # Original Image
    # -----------------------------
    with col1:
        st.subheader("📸 Original Image")
        st.image(image, use_column_width=True)

    # -----------------------------
    # SAHI Prediction
    # -----------------------------
    with st.spinner("🔍 Detecting roses..."):
        result = get_sliced_prediction(
            image_np,
            detection_model,
            slice_height=384,
            slice_width=384,
            overlap_height_ratio=0.35,
            overlap_width_ratio=0.35,
        )

    boxes = result.object_prediction_list

    # -----------------------------
    # Smart Filtering
    # -----------------------------
    filtered_boxes = []

    for obj in boxes:
        bbox = obj.bbox

        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)

        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area > 1200:  # remove noise
            filtered_boxes.append((x1, y1, x2, y2))

    rose_count = len(filtered_boxes)

    # -----------------------------
    # Draw Boxes
    # -----------------------------
    for (x1, y1, x2, y2) in filtered_boxes:
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # -----------------------------
    # Result Image
    # -----------------------------
    with col2:
        st.subheader("🎯 Detected Roses")
        st.image(image_np, use_column_width=True)

    st.markdown("---")

    # -----------------------------
    # Metrics Display
    # -----------------------------
    m1, m2, m3 = st.columns(3)

    m1.metric("🌹 Total Roses", rose_count)
    m2.metric("📦 Raw Detections", len(boxes))
    m3.metric("✅ Filtered", len(filtered_boxes))

    # -----------------------------
    # Footer Note
    # -----------------------------
    st.info("💡 Small detections (like leaves) are filtered using area threshold.")