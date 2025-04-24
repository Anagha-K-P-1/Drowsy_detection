import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(page_title="Drowsiness Detector", page_icon="😴", layout="wide")

# Load Trained Model with Caching
@st.cache_resource
def load_drowsy_model():
    try:
        model = load_model("drowsy_detection.keras")
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Image Preprocessing
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img)

    if len(img_array.shape) == 3 and img_array.shape[2] > 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])

    return img_array

# Prediction Function
def predict_drowsiness(image, model):
    if model is None:
        return None

    img_array = preprocess_image(image)

    try:
        prediction = model.predict(img_array)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

    class_idx = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    class_names = ['DROWSY','NATURAL']

    return {
        'class': class_names[class_idx],
        'confidence': confidence * 100
    }

# Main App
def main():
    st.title("😴 Drowsiness Detection")

    model = load_drowsy_model()

    input_method = st.radio("Select Input Method", ["Upload Image", "Camera Capture"])

    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload a facial or eye image", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_input = st.camera_input("Take a picture")
        if camera_input:
            image = Image.open(camera_input).convert('RGB')

    if image:
        st.image(image, width=300, caption="Input Image")

        if st.button("Check Drowsiness"):
            with st.spinner("Analyzing..."):
                result = predict_drowsiness(image, model)

                if result:
                    status = result['class']
                    confidence = result['confidence']
                    if status == 'Drowsy':
                        st.error(f"Status: **{status}** with {confidence:.2f}% confidence.")
                        st.warning("⚠️ Please take a break or rest if you're driving.")
                    else:
                        st.success(f"Status: **{status}** with {confidence:.2f}% confidence.")
                else:
                    st.warning("Could not make a prediction. Try a clearer image.")

if __name__ == "__main__":
    main()