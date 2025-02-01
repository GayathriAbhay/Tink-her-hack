import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load trained AI model
model = load_model('outfit_model.h5')

# Define class labels for the predictions
class_labels = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# Function to preprocess uploaded image
def prepare_image(image):
    image = image.resize((28, 28))  # Resize to 28x28
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input
    return image

# Function to make predictions
def predict(image):
    image = prepare_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability
    return class_labels[predicted_class]

# Streamlit UI
st.title("👗 AI Outfit Suggestion App")
st.write("Upload an outfit image, and the AI will predict its category!")

# File uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict and display result
    predicted_category = predict(img)
    st.success(f"Predicted Outfit Category: **{predicted_category}**")

# Load the trained models
outfit_model = load_model('outfit_model.h5')
accessory_model = load_model('accessory_model.h5')

def recommend_accessories(image):
    # Preprocess the accessory image (resize and normalize)
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = image.reshape(1, 64, 64, 3)

    # Get accessory prediction
    prediction = accessory_model.predict(image)
    accessory_classes = ['Shoes', 'Bag', 'Jewelry', 'Hat']  # Modify according to your dataset
    predicted_accessory = accessory_classes[np.argmax(prediction)]

    return predicted_accessory

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    # Outfit Prediction
    outfit_prediction = predict_outfit(image)
    st.write(f"Predicted Outfit: {outfit_prediction}")

    # Accessory Prediction
    accessory_recommendation = recommend_accessories(image)
    st.write(f"Suggested Accessory: {accessory_recommendation}")
