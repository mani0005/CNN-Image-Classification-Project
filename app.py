import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import base64
from PIL import Image
import io

# ---------- Background Image ----------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("/Users/apple/Desktop/fashion_classification_project/bg.jpeg")

# ---------- Load Model ----------
model = load_model("/Users/apple/Desktop/CNN_MODEL/model.h5")

# ---------- Load Labels ----------
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(
    "/Users/apple/Desktop/CNN_MODEL/label.npy",
    allow_pickle=True
)

# ---------- Image Preprocessing ----------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------- Prediction ----------
def predict_fashion_category(img_array):
    result = model.predict(img_array)
    predicted_class = np.argmax(result)
    return label_encoder.inverse_transform([predicted_class])[0]

# ---------- Streamlit App ----------
def main():
    st.markdown(
        """
        <style>
        .title {
            font-family: "Monotype Corsiva", cursive;
            color: brown;
            font-size: 80px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="title">Style Scan</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a fashion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        img_array = preprocess_image(uploaded_file)
        predicted_label = predict_fashion_category(img_array)

        st.markdown(
            f"<h2 style='color:brown;'>Predicted Fashion Category: {predicted_label}</h2>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()

