import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model_path = 'C:/Users/FPTB-ICT-MIS/Desktop/Nasu Project/PPPP/Leukemia Detection using Deep Learning/leukemia_classification.h5'  # Replace with your model path
model = tf.keras.models.load_model(model_path)

# Define class labels (modify as per your model's classes)
class_labels = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the size your model expects
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Use the preprocessing function you used during training
    return image

# Function to make predictions
def make_prediction(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return class_labels[np.argmax(predictions)]

# Streamlit app
st.title("Leukemia Detection using Deep Learning")

st.write("Upload an image to classify it as Benign or Malignant.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = make_prediction(image)
    st.write(f"Prediction: {prediction}")
