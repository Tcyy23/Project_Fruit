from flask import Flask, request, render_template
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
# fruit_classification_model.keras - 5 classes of fruit from one data using tensorflow (100*100)
# fruit_classification_model100p.pth - 100 classes of fruit from one data set (100*100)
# fruit_classification_model2.pth - 100 classes of fruit merging from two different data sets (224*224)
# fruit_classification_model3.pth - 53 classes of fruit merging from three different data sets (224*224) (More common fruit)

model = torch.load('fruit_classification_model2.pth', map_location=torch.device('cpu'))

model.eval()  # Set the model to evaluation mode

# Define the class names
classFile = 'Fruit/classname.txt'
class_names = []

# Open the file in read mode
with open(classFile, 'r') as file:
    for line in file:
        class_names.append(line.strip())


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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(filename).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Make a prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top3_prob, top3_catid = torch.topk(probabilities, 3)

            predictions = [(class_names[catid], prob.item()) for prob, catid in zip(top3_prob, top3_catid)]

        # Prepare the results for rendering
        results = '\n'.join([f'{pred[0]}: {pred[1]:.4f}' for pred in predictions])

        return f'The top 3 predictions are:\n{results}'


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
