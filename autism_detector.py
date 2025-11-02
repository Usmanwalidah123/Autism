import streamlit as st
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications import ResNet50
from keras.applications import resnet
import numpy as np
from PIL import Image
import os

# --- Constants ---
# This MUST match the file you saved from your Colab script
WEIGHTS_FILE = 'autism_resnet50_weights.weights.h5' 
IMAGE_SIZE = 224
# These must match the order your model was trained on
# After sorting, 'Autistic' is 0, 'Non-Autistic' is 1
CLASS_NAMES = ['Autistic', 'Non-Autistic'] 
NUM_CLASSES = 2

# --- Model Definition ---

def build_model():
    """
    Builds the exact same model architecture as in the training script.
    Note: 'weights=None' because we are loading our own saved weights.
    """
    base_model_resnet = ResNet50(include_top=False, 
                                 weights=None,  # We will load our own weights
                                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # We don't need to set trainable flags for inference, but
    # the architecture must be identical.
    
    model = Sequential([
        base_model_resnet,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'), # From your config
        Dropout(0.4),                 # From your config
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

@st.cache_resource
def load_model():
    """
    Loads the model and weights. Caches the loaded model.
    """
    try:
        model = build_model()
        model.load_weights(WEIGHTS_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.error(f"Please make sure the file '{WEIGHTS_FILE}' is in the same directory as this script.")
        return None

def preprocess_image(image_pil):
    """
    Preprocesses the uploaded PIL image to be ready for the model.
    """
    # Resize
    image_resized = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to numpy array
    image_array = tf.keras.utils.img_to_array(image_resized)
    
    # Handle RGBA images (e.g., from PNGs) by dropping the alpha channel
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
        
    # Expand dimensions to create a batch of 1
    image_batch = tf.expand_dims(image_array, 0)
    
    # Apply the ResNet-specific preprocessing
    processed_image = resnet.preprocess_input(image_batch)
    
    return processed_image

# --- Streamlit UI ---

st.set_page_config(page_title="Autism Detection", layout="centered")
st.title("🧠 Autism Spectrum Disorder (ASD) Image Detector")
st.write("This app uses a deep learning model (ResNet50) trained on the AutismDataset to predict whether a provided image is in the 'Autistic' or 'Non-Autistic' class.")
st.write(f"**Note:** This is a demo tool based on your trained model. Please ensure the `{WEIGHTS_FILE}` file is present in the app's root directory.")

# Check if weights file exists
if not os.path.exists(WEIGHTS_FILE):
    st.error(f"**Model weights file not found!**")
    st.write(f"Please download your trained weights file, rename it to `{WEIGHTS_FILE}`, and place it in the same folder as this Streamlit app.")
    st.stop()

# Load the model
model = load_model()

if model is not None:
    st.success("✅ Model loaded successfully!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open the image
            image = Image.open(uploaded_file)

            # Display the image
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")

            with st.spinner('🤖 Analyzing the image...'):
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Make prediction
                prediction = model.predict(processed_image)
                
                # Get the class and confidence
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction) * 100
                
            # Display the result
            st.subheader("Prediction Result")
            if predicted_class_name == 'Autistic':
                st.markdown(f"## <span style='color:orange;'>Prediction: {predicted_class_name}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"## <span style='color:green;'>Prediction: {predicted_class_name}</span>", unsafe_allow_html=True)
                
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Display detailed probabilities
            with st.expander("Show detailed probabilities"):
                st.write(f"**{CLASS_NAMES[0]} (Autistic):** {prediction[0][0]*100:.2f}%")
                st.write(f"**{CLASS_NAMES[1]} (Non-Autistic):** {prediction[0][1]*100:.2f}%")
        
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
