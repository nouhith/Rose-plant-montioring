import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
from model import CSRNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# NORMALIZATION (CSRNet)
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():

    yolo = YOLO("Yield_model.pt")

    density = CSRNet()
    density.load_state_dict(
        torch.load("best_density_model (2).pth", map_location=device)
    )

    density.to(device)
    density.eval()

    return yolo, density


yolo_model, density_model = load_models()


# -------------------------
# CLAHE
# -------------------------
def apply_clahe(img):

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


# -------------------------
# CENTER CROP
# -------------------------
def center_crop(img):

    h,w,_ = img.shape

    size = min(h,w)

    startx = w//2 - size//2
    starty = h//2 - size//2

    return img[starty:starty+size, startx:startx+size]


# -------------------------
# YOLO TTA
# -------------------------
def yolo_tta(img,conf):

    imgs = [
        img,
        cv2.flip(img,1),
        cv2.flip(img,0)
    ]

    counts = []

    for im in imgs:

        res = yolo_model(im,conf=conf)

        counts.append(len(res[0].boxes))

    count = int(np.mean(counts))

    result_img = yolo_model(img,conf=conf)[0].plot()

    return count,result_img


# -------------------------
# DENSITY INFERENCE + TTA
# -------------------------
def density_inference(img):

    scales = [0.75,1.0,1.25]

    aug_imgs = [
        img,
        cv2.flip(img,1),
        cv2.flip(img,0),
        cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    ]

    counts = []
    maps = []

    for im in aug_imgs:

        for s in scales:

            resized = cv2.resize(im,(0,0),fx=s,fy=s)
            resized = cv2.resize(resized,(640,640))

            tensor = transform(resized).unsqueeze(0).to(device)

            with torch.no_grad():

                pred = density_model(tensor)

            density = pred[0][0].cpu().numpy()

            counts.append(density.sum())
            maps.append(density)

    final_count = int(np.mean(counts))
    final_map = np.mean(maps,axis=0)

    return final_count,final_map


# -------------------------
# STREAMLIT UI
# -------------------------
st.title("🌹 Smart Rose Counting System")

method = st.sidebar.selectbox(
    "Method",
    ["YOLO Detection","Density Counting","Auto Mode"]
)

confidence = st.sidebar.slider(
    "YOLO Confidence",
    0.1,1.0,0.4
)

use_clahe = st.sidebar.checkbox("CLAHE Enhancement",True)

use_crop = st.sidebar.checkbox("Center Crop",True)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    if use_crop:
        img = center_crop(img)

    if use_clahe:
        img = apply_clahe(img)


    col1,col2 = st.columns(2)


    with col1:
        st.image(img,caption="Processed Image",use_container_width=True)


# -------------------------
# YOLO MODE
# -------------------------
    if method == "YOLO Detection":

        count,result = yolo_tta(img,confidence)

        with col2:
            st.image(result,use_container_width=True)

        st.success(f"🌹 Roses Detected: {count}")


# -------------------------
# DENSITY MODE
# -------------------------
    elif method == "Density Counting":

        count,density_map = density_inference(img)

        with col2:

            fig,ax = plt.subplots()

            ax.imshow(density_map,cmap="jet")

            ax.set_title("Density Map")

            st.pyplot(fig)

        st.success(f"🌹 Estimated Roses: {count}")


# -------------------------
# AUTO MODE
# -------------------------
    else:

        yolo_count,result = yolo_tta(img,confidence)

        if yolo_count > 20:

            count,density_map = density_inference(img)

            with col2:

                fig,ax = plt.subplots()

                ax.imshow(density_map,cmap="jet")

                st.pyplot(fig)

            st.success(f"🌹 Auto Count (Density): {count}")

        else:

            with col2:
                st.image(result,use_container_width=True)

            st.success(f"🌹 Auto Count (YOLO): {yolo_count}")