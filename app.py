import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess
)

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)

from PIL import Image

# ==================================
# TITLE
# ==================================

st.title("Fake vs Real Image Detector")

# ==================================
# MODEL SELECTION
# ==================================

model_choice = st.selectbox(
    "Select Model",
    [
        "CNN",
        "ResNet50",
        "EfficientNetB0"
    ]
)

# ==================================
# LOAD MODEL
# ==================================

if model_choice == "CNN":

    model = load_model(
        r"E:\Capstone 1\cnn_model.h5"
    )

    image_size = (128, 128)

elif model_choice == "ResNet50":

    model = load_model(
        r"E:\Capstone 1\resnet50_model.h5"
    )

    image_size = (224, 224)

else:

    model = load_model(
        r"E:\Capstone 1\efficientnet_model.h5"
    )

    image_size = (224, 224)

# ==================================
# IMAGE UPLOAD
# ==================================

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

# ==================================
# PREDICTION
# ==================================

if uploaded_file is not None:

    img = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize according to model

    img = img.resize(
        image_size
    )

    img_array = np.array(
        img
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ==========================
    # MODEL-SPECIFIC PREPROCESS
    # ==========================

    if model_choice == "CNN":

        img_array = img_array / 255.0

    elif model_choice == "ResNet50":

        img_array = resnet_preprocess(
            img_array
        )

    else:

        img_array = efficientnet_preprocess(
            img_array
        )

    # ==========================
    # PREDICT
    # ==========================

    prediction = model.predict(
        img_array,
        verbose=0
    )

    confidence = prediction[0][0]

    if confidence > 0.5:

        result = "Real"

        conf = confidence * 100

    else:

        result = "Fake"

        conf = (1 - confidence) * 100

    # ==========================
    # DISPLAY RESULT
    # ==========================

    st.success(
        f"Prediction: {result}"
    )

    st.write(
        f"Confidence: {conf:.2f}%"
    )

    st.info(
        f"Model Used: {model_choice}"
    )