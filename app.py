import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load trained AI models
outfit_model = load_model('outfit_model.h5')
accessory_model = load_model('accessory_model.h5')

# Define class labels for the predictions
outfit_classes = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

accessory_classes = ['Shoes', 'Bag', 'Jewelry', 'Hat']  # Modify based on dataset

# Function to preprocess outfit image
def prepare_outfit_image(image):
    image = image.resize((28, 28))  # Resize to 28x28
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input
    return image

# Function to make outfit predictions
def predict_outfit(image):
    image = prepare_outfit_image(image)
    prediction = outfit_model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability
    return outfit_classes[predicted_class]

# Function to preprocess accessory image
def prepare_accessory_image(image):
    image = image.resize((64, 64))  # Resize to 64x64
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 64, 64, 3)  # Reshape for CNN input
    return image

# Function to recommend accessories
def recommend_accessories(image):
    image = prepare_accessory_image(image)
    prediction = accessory_model.predict(image)
    predicted_accessory = accessory_classes[np.argmax(prediction)]
    return predicted_accessory

# Streamlit UI
st.title("👗 AI Outfit & Accessory Recommendation App")
st.write("Upload an outfit image, and the AI will predict its category and suggest accessories!")

# File uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Outfit Prediction
    outfit_prediction = predict_outfit(img)
    st.success(f"Predicted Outfit Category: **{outfit_prediction}**")

    # Accessory Prediction
    accessory_recommendation = recommend_accessories(img)
    st.info(f"Suggested Accessory: **{accessory_recommendation}**")
