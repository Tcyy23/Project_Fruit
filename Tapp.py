from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('fruit_classification_model.keras')

# Define the class names
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        # Save the file to a temporary directory
        filename = os.path.join('uploads', file.filename)
        file.save(filename)

        # Preprocess the image
        img = Image.open(filename).convert('RGB').resize((100, 100))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 100, 100, 3)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions[0])
        predicted_class = class_names[predicted_label]

        return f'The image is a {predicted_class}.'

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
