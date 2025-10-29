import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import resnet
import numpy as np
from PIL import Image
import io
import os

# --- Configuration and Constants ---
IMAGE_SIZE = 224
CLASS_NAMES = ['Autistic', 'Non_Autistic']
WEIGHTS_FILE = 'autism_resnet_weights.h5' # IMPORTANT: Ensure this file is in the same directory!

# --- Model Loading and Caching ---
@st.cache_resource
def load_model():
    """Builds and loads the structure and weights of the fine-tuned ResNet50 model."""
    st.write(f"Attempting to load model from {WEIGHTS_FILE}...")
    try:
        # 1. Define the exact base model structure (Frozen initially)
        base_model_resnet = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
        )
        base_model_resnet.trainable = True # Set to True to match the fine-tuned state

        # 2. Define the exact head structure
        model = Sequential([
            base_model_resnet,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(len(CLASS_NAMES), activation='softmax')
        ])

        # 3. Load the pre-trained weights
        if not os.path.exists(WEIGHTS_FILE):
             st.error(f"Error: Model weights file '{WEIGHTS_FILE}' not found.")
             st.error("Please ensure you have trained the model and saved the weights to this filename.")
             return None

        model.load_weights(WEIGHTS_FILE)
        st.success("Model weights loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load the model or weights: {e}")
        return None

# --- Prediction Function ---
def predict(model, uploaded_file):
    """
    Preprocesses the uploaded image and makes a prediction.
    """
    # Load and convert image to a numpy array
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image)

    # Convert to a 4D tensor (batch size 1)
    image_tensor = tf.expand_dims(image_array, axis=0)

    # Apply the ResNet-specific preprocessing
    # This scales pixel values according to the ImageNet dataset's statistics
    processed_tensor = resnet.preprocess_input(image_tensor)

    # Make prediction
    predictions = model.predict(processed_tensor)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return CLASS_NAMES[predicted_class_index], confidence, image

# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="Autism Spectrum Disorder (ASD) Image Classification",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
            .main-header {font-size: 2.5em; color: #1f77b4;}
            .subheader {font-size: 1.5em; color: #ff7f0e;}
            .stFileUploader {padding-top: 10px;}
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 8px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🧠 Autism Detection from Facial Images</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload a facial image (224x224 recommended) for classification using a fine-tuned ResNet50 model.</p>', unsafe_allow_html=True)

    # Load the model only once
    with st.spinner('Loading Deep Learning Model...'):
        model = load_model()

    if model is None:
        st.error("Application cannot run without the model weights file. Please check the console/logs.")
        return

    st.sidebar.header("Image Uploader")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model Status:** Ready")
    st.sidebar.markdown(f"**Target Classes:** {', '.join(CLASS_NAMES)}")


    col1, col2 = st.columns([1, 1])

    with col1:
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            if st.button('Classify Image', key='classify_btn'):
                with st.spinner('Analyzing image...'):
                    # Perform prediction
                    predicted_class, confidence, original_image = predict(model, uploaded_file)
                    probability = confidence * 100

                with col2:
                    st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)

                    if predicted_class == 'Autistic':
                        result_color = '#d62728' # Red
                    else:
                        result_color = '#2ca02c' # Green

                    st.markdown(f"""
                        <div style="
                            background-color: {result_color};
                            padding: 20px;
                            border-radius: 10px;
                            color: white;
                            text-align: center;
                            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                        ">
                            <h3>Predicted Class: {predicted_class}</h3>
                            <p style="font-size: 1.2em;">Confidence: {probability:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.info("""
                        **Disclaimer:** This tool is a deep learning model for research and demonstration purposes only.
                        It is **not a substitute for professional medical diagnosis**. Always consult a qualified
                        healthcare professional for any health concerns or diagnostic information.
                    """)
            else:
                with col2:
                    st.warning("Click 'Classify Image' to run the model.")
        else:
            st.info("Please upload an image in the sidebar to begin.")

if __name__ == '__main__':
    # Streamlit requires a function to execute the app logic
    main()
