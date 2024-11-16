import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the previously trained model
model_path = 'C:\\Users\\Danie\\OneDrive\\Desktop\\AER 815 Codes\\Daniel-Ng-AER850-Repository\\Project 2\\model.h5'  
model = load_model(model_path)
class_indices_path = 'C:\\Users\\Danie\\OneDrive\\Desktop\\AER 815 Codes\\Daniel-Ng-AER850-Repository\\Project 2\\class_indices.json'
with open(class_indices_path, 'r') as class_file:
    class_indices = json.load(class_file)

# Function to process and predict a single image
def processing(img_path):
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255. 
    return img_array

# Load the image file, target size must match the input size of the model
def predicting(img_path, model):
    img_array = processing(img_path)
    stat = model.predict(img_array)
    return stat

test_image_path = {
    'crack': 'C:\\Users\\Danie\\OneDrive\\Desktop\\AER 815 Codes\\Daniel-Ng-AER850-Repository\\Project 2\\Project 2 Data\\Data\\test\\crack\\test_crack.jpg',
    'missing-head': 'C:\\Users\\Danie\\OneDrive\\Desktop\\AER 815 Codes\\Daniel-Ng-AER850-Repository\\Project 2\\Project 2 Data\\Data\\test\\missing-head\\test_missinghead.jpg',
    'paint-off': 'C:\\Users\\Danie\\OneDrive\\Desktop\\AER 815 Codes\\Daniel-Ng-AER850-Repository\\Project 2\\Project 2 Data\\Data\\test\\paint-off\\test_paintoff.jpg',
}

# Plot the image with predictions
for true_label, img_path in test_image_path.items():
    image_name = os.path.basename(img_path)
    print(f"Processing image: {image_name}") 

    probabilities = predicting(img_path, model)[0]
    predicted_label = np.argmax(probabilities)
    predicted_label_name = list(class_indices.keys())[predicted_label]  

    img = image.load_img(img_path) 
    plt.figure(figsize=(6, 8))  
    plt.imshow(img)
    plt.axis('off') 

    # Set the title with the classification label
    plt.title(f"True Label: {true_label} | Predicted Label: {predicted_label_name}", color='black', fontsize=14)

    # Overlay the prediction percentages on the image
    prediction_text = '\n'.join([f'{label}: {prob:.2%}' for label, prob in zip(class_indices.keys(), probabilities)])
    plt.text(10, img.size[1] - 20, prediction_text, color='#39FF14', fontsize=12, weight='bold')

    plt.show()

