from flask import Flask, request, render_template, jsonify
import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the plant classes and prevention measures
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

# Function to create a model
def create_model(num_classes):
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# Load models dynamically
def load_combined_models(filepath):
    combined_models = torch.load(filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    models_per_plant = {}
    for plant, state_dict in combined_models.items():
        num_classes = len(state_dict["classifier.1.bias"])
        model = create_model(num_classes)
        model.load_state_dict(state_dict)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        models_per_plant[plant] = model
    return models_per_plant

# Load combined models once
combined_model_path = "combined_models.pth"
models_per_plant = load_combined_models(combined_model_path)

# Prediction function
def predict_disease_combined(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    best_prediction = None

    for plant, model in models_per_plant.items():
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_class_index = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][predicted_class_index].item()

            if not best_prediction or confidence > best_prediction["confidence"]:
                best_prediction = {
                    "plant": plant,
                    "class_index": predicted_class_index.item(),
                    "confidence": confidence
                }
    
    if best_prediction:
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
    else:
        return {"Error": "No prediction available."}

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
        
        result = predict_disease_combined(filepath)
        os.remove(filepath)  # Optional: Remove file after processing
        
        return jsonify(result)

if __name__ == '__main__':
    app.run()


