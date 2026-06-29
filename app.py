import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Fake vs Real Image Detector",
    page_icon="🤖",
    layout="centered"
)

# ==========================================================
# TITLE
# ==========================================================

st.title("🤖 Fake vs Real Image Detection")
st.markdown(
    """
This application uses **Deep Learning** models to detect whether an uploaded face image is **Real** or **Fake**.

### Available Models
- CNN
- ResNet50
- EfficientNetB0
"""
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    (
        "CNN",
        "ResNet50",
        "EfficientNetB0"
    )
)

# ==========================================================
# LOAD MODEL (CACHE)
# ==========================================================

@st.cache_resource
def load_selected_model(choice):

    if choice == "CNN":
        model = load_model("cnn_model.h5")
        image_size = (128, 128)

    elif choice == "ResNet50":
        model = load_model("resnet50_model.h5")
        image_size = (224, 224)

    else:
        model = load_model("efficientnet_model.h5")
        image_size = (224, 224)

    return model, image_size


model, image_size = load_selected_model(model_choice)

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload a Face Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize(image_size)

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    # --------------------------------------

    if model_choice == "CNN":

        img_array = img_array / 255.0

    elif model_choice == "ResNet50":

        img_array = resnet_preprocess(img_array)

    else:

        img_array = efficientnet_preprocess(img_array)

    # --------------------------------------

    with st.spinner("Analyzing Image..."):

        prediction = model.predict(
            img_array,
            verbose=0
        )

    confidence = float(prediction[0][0])

    if confidence > 0.5:

        result = "Real"

        confidence_score = confidence * 100

        st.success("✅ Prediction: REAL")

    else:

        result = "Fake"

        confidence_score = (1 - confidence) * 100

        st.error("❌ Prediction: FAKE")

    # ======================================================

    st.write("### Confidence")

    st.progress(min(confidence_score / 100, 1.0))

    st.write(f"**Confidence Score:** {confidence_score:.2f}%")

    st.write(f"**Model Used:** {model_choice}")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
### Capstone Project

**Fake vs Real Image Detection Using Deep Learning**

**Models Implemented**
- CNN
- ResNet50
- EfficientNetB0

Developed using **TensorFlow**, **Keras**, **Python**, and **Streamlit**.
"""
)
