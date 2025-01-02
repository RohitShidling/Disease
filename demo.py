import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
import os
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model and disease data
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

# Load model
def load_combined_models(filepath):
    logging.info("Loading combined model...")
    return torch.jit.load(filepath)

combined_model_path = "combined_models.ptl"
combined_model = load_combined_models(combined_model_path)

# Prediction function
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if isinstance(outputs, (tuple, list)):
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

# Routes
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
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = predict_disease_combined(filepath)
            return jsonify(result)
        except UnidentifiedImageError:
            logging.error("Invalid image file.")
            return jsonify({"Error": "Uploaded file is not a valid image."})
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({"Error": "An internal error occurred."})
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

# Main entry point
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # For server hosting
    app.run(host="0.0.0.0", port=port)
