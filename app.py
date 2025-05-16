import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import io

app = Flask(__name__)

# Load your trained model
model_path = 'soil_classification_model_resnet18.pth'
model = None
model_name = 'resnet18'


def load_model():
    global model, model_name
    if model is None:
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 4)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 4)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        model.load_state_dict(torch.load(model_path))
        model.eval()
    return model


# Preprocess the image
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image


# Make a prediction
def predict_soil_type(image_bytes):
    model = load_model()
    if model is None:
        return "Error: Model not loaded"
    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = transform_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    class_names = ['alluvial soil', 'black soil', 'clay soil', 'red soil']
    predicted_soil = class_names[predicted_class.item()]
    return predicted_soil


def get_crop_recommendation(soil_type):
    if soil_type == 'alluvial soil':
        return ['Rice', 'Wheat', 'Sugarcane', 'Jute']
    elif soil_type == 'black soil':
        return ['Cotton', 'Sugarcane', 'Groundnut', 'Tobacco']
    elif soil_type == 'clay soil':
        return ['Rice', 'Wheat', 'Cotton', 'Sugarcane']
    elif soil_type == 'red soil':
        return ['Groundnut', 'Millet', 'Cotton', 'Tobacco']
    else:
        return ['No suitable crops found']


@app.route('/recommend', methods=['POST'])
def recommend():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    try:
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))  # Open the image
        image = image.resize((224, 224))  # Resize the image  <---- Add this line

        soil_type = predict_soil_type(image_bytes)
        crops = get_crop_recommendation(soil_type)
        response = {
            'soil_type': soil_type,
            'crops': crops
        }
        return jsonify(response), 200

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': f'Error processing image: {e}'}), 500



if __name__ == '__main__':
    load_model()
    app.run(debug=True)
