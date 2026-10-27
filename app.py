
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Face Mask Detection | MaskAI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROFESSIONAL DARK THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #111c30);
    color: #f1f5f9;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background-color: #0a1020;
    border-right: 1px solid #263449;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
}

p, label {
    color: #cbd5e1;
}

[data-testid="stMetric"] {
    background-color: #172338;
    border: 1px solid #2b3b52;
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stFileUploader"] {
    background-color: #172338;
    border: 1px dashed #3b536e;
    border-radius: 12px;
    padding: 12px;
}

hr {
    border-color: #2b3b52;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_mask_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = (128, 128)

DISPLAY_NAMES = {
    "with_mask": "Mask Worn Correctly",
    "without_mask": "No Mask",
    "incorrect_mask": "Mask Worn Incorrectly"
}


# =====================================================
# 4. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# =====================================================
# 5. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):
    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        resample=Image.Resampling.NEAREST
    )

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:
    st.title("😷 MaskAI")
    st.caption("Face Mask Detection System")

    st.divider()

    st.subheader("Project Overview")
    st.write("**Algorithm:** CNN")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Input Size:** 128 × 128 RGB")
    st.write("**Output Classes:** 3")

    st.divider()

    st.subheader("Prediction Categories")
    st.markdown("🟢 **Mask Worn Correctly**")
    st.markdown("🔴 **Mask Worn Incorrectly**")
    st.markdown("🟠 **No Mask**")

    st.divider()

    st.subheader("How to Use")
    st.write("1. Upload a face image.")
    st.write("2. View the prediction.")
    st.write("3. Compare class probabilities.")

    st.divider()
    st.caption("AI/ML Portfolio Project")


# =====================================================
# 7. HEADER — NATIVE STREAMLIT, NO HTML
# =====================================================



st.title("😷 Face Mask Detection")

st.markdown(
    "An AI-powered image classification application that identifies "
    "correctly worn masks, incorrectly worn masks, and faces without "
    "masks using a trained Convolutional Neural Network (CNN)."
)

st.divider()


# =====================================================
# 8. CHECK MODEL FILES
# =====================================================

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not CLASS_NAMES_PATH.exists():
    st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error("Unable to load the model or class names.")
    st.exception(error)
    st.stop()


# =====================================================
# 9. IMAGE UPLOAD
# =====================================================

st.header(" Upload & Analyze")

st.write(
    "Upload a clear face image to classify mask usage."
)

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP."
)


# =====================================================
# 10. PREDICTION
# =====================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        image_array = preprocess_image(image)

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]

        confidence = float(predictions[predicted_index]) * 100

        display_name = DISPLAY_NAMES.get(
            predicted_class,
            predicted_class
        )

        if predicted_class == "with_mask":
            result_icon = "🟢"
        elif predicted_class == "incorrect_mask":
            result_icon = "🔴"
        else:
            result_icon = "🟠"

        st.divider()

        image_col, result_col = st.columns(
            [1.1, 1],
            gap="large"
        )

        with image_col:
            st.subheader("Uploaded Image")

            st.image(
                image,
                caption="Image selected for analysis",
                width="stretch"
            )

            st.caption(
                f"Image dimensions: {image.width} × {image.height}"
            )

        with result_col:
            st.subheader("Prediction Result")

            st.success(
                f"{result_icon} Predicted Category: {display_name}"
            )

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the prediction is correct."
            )

        # =============================================
        # CLASS PROBABILITIES
        # =============================================

        st.divider()

        st.header("📊 Class Probability Distribution")

        st.write(
            "Compare the predicted probability for each category."
        )

        for index, class_name in enumerate(class_names):
            probability = float(predictions[index]) * 100

            label = DISPLAY_NAMES.get(
                class_name,
                class_name
            )

            st.write(f"**{label}** — {probability:.2f}%")

            st.progress(
                min(max(probability / 100, 0.0), 1.0)
            )

        # =============================================
        # TECHNICAL DETAILS
        # =============================================

        with st.expander("View Technical Details"):
            st.write("Model path:", str(MODEL_PATH))
            st.write("Class names:", class_names)
            st.write("Predicted index:", predicted_index)
            st.write("Processed image shape:", image_array.shape)
            st.write("Raw probabilities:", predictions.tolist())

    except Exception as error:
        st.error("An error occurred while processing the image.")
        st.exception(error)


# =====================================================
# 11. EMPTY STATE
# =====================================================

else:
    st.info(
        "Upload an image above to start face mask classification."
    )

    st.header("About This Application")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🟢 Correctly Worn")
        st.write(
            "Predicts whether a face image belongs to the "
            "correctly worn mask category."
        )

    with col2:
        st.subheader("🔴 Incorrectly Worn")
        st.write(
            "Predicts whether a face image belongs to the "
            "incorrectly worn mask category."
        )

    with col3:
        st.subheader("🟠 No Mask")
        st.write(
            "Predicts whether a face image belongs to the "
            "no-mask category."
        )


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

st.caption(
    "❤️ Face Mask Detection  |  CNN  |  TensorFlow  |  Streamlit"
)
