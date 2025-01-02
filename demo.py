import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
from torchvision import transforms, models

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

plant_classes = {
    "corn": ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"],
    "rice": ["Bacterial leaf blight", "Brown spot", "Leaf smut"],
    "sorghum": ["Anthracnose and Red Rot", "Cereal Grain molds", "Covered Kernel smut", 
                "Head Smut", "Rust", "loose smut"],
    "wheat": ["Healthy", "septoria", "stripe_rust"]
}

preventions = {
    "Bacterial leaf blight": "Apply appropriate antibiotics and improve water management.",
    "Brown spot": "Use resistant varieties and apply fungicides if necessary.",
    "Leaf smut": "Use resistant varieties and reduce nitrogen application.",
    "septoria": "Use fungicide sprays and avoid overhead irrigation.",
    "stripe_rust": "Apply fungicides and plant resistant varieties.",
    "Blight": "Remove infected plants and use fungicide sprays.",
    "Common_Rust": "Use resistant corn hybrids and rotate crops.",
    "Gray_Leaf_Spot": "Remove infected leaves, apply fungicides, and plant disease-resistant varieties.",
    "Anthracnose and Red Rot": "Remove infected parts, apply fungicides, and rotate crops.",
    "Cereal Grain molds": "Use fungicides and plant mold-resistant varieties.",
    "Covered Kernel smut": "Use certified smut-free seed and apply seed treatments.",
    "Head Smut": "Use resistant varieties and apply fungicides.",
    "Rust": "Apply fungicides and use resistant varieties.",
    "loose smut": "Use certified disease-free seeds and seed treatments.",
    "Healthy": "No action needed."
}

def load_combined_models(filepath):
    return torch.jit.load(filepath)

combined_model_path = "combined_models.ptl"
combined_model = load_combined_models(combined_model_path)

def predict_disease_combined(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run prediction
    outputs = combined_model(img_tensor)
    
    if isinstance(outputs, tuple):
        # Multiple outputs (one per plant type)
        best_prediction = None
        for idx, plant_output in enumerate(outputs):
            confidence, predicted_class = torch.max(torch.softmax(plant_output, dim=1), dim=1)
            confidence = confidence.item()
            if not best_prediction or confidence > best_prediction["confidence"]:
                plant = list(plant_classes.keys())[idx]
                best_prediction = {
                    "plant": plant,
                    "class_index": predicted_class.item(),
                    "confidence": confidence
                }
    else:
        # Single output
        confidence, predicted_class = torch.max(torch.softmax(outputs, dim=1), dim=1)
        plant = list(plant_classes.keys())[0]
        best_prediction = {
            "plant": plant,
            "class_index": predicted_class.item(),
            "confidence": confidence.item()
        }

    plant = best_prediction["plant"]
    class_index = best_prediction["class_index"]
    predicted_class = plant_classes[plant][class_index]
    prevention = preventions.get(predicted_class, "No specific prevention available.")
    
    return {
        "Plant": plant.capitalize(),
        "Disease": predicted_class,
        "Confidence": round(best_prediction["confidence"], 4),
        "Prevention": prevention
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"Error": "No file part"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"Error": "No selected file"})
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            result = predict_disease_combined(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({"Error": str(e)})
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run()