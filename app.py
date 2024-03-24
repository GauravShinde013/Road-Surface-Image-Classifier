import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model(r'D:\MSc AI\SEM2\Research\A2\model.h5')

# Function to predict the class of an image
def predict_class(img, model):
    # Preprocess the image to match the input shape of the model
    img = img.resize((224, 224))  # Replace with your model's expected input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale pixel values

    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    return class_idx

# Streamlit app
st.title('Road Surface Image Classifier')

# File selection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image', width=300)
    st.write("")
    st.write("Classifying...")

    # Open and predict the class of the image
    img = Image.open(uploaded_file)
    class_idx = predict_class(img, model)
    
    # Convert the index to the respective class name (change the names as per your dataset)
    class_names = ['water_asphalt_slight', 'water_concrete_slight', 'water_gravel']
    class_name = class_names[class_idx]

    # Show the result
    st.success(f"The image is classified as: {class_name}")
