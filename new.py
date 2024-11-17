from flask import Flask, request, jsonify, send_file
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import io

app = Flask(__name__)

# Load your custom-trained ViT model and feature extractor
model_name = "google/vit-base-patch16-224-in21k"  # Replace with the path to your custom-trained model
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Check if CUDA is available for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image preprocessing function
def preprocess_image(image):
    """  
    Preprocess the input image for the ViT model.
    """
    image = image.convert("RGB")  # Ensure image is in RGB format
    inputs = feature_extractor(images=image, return_tensors="pt")  # Preprocess image
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU if available
    return inputs

# Prediction function to classify keratoconus or normal
def predict_keratoconus(image):
    """
    Predict if the image is of keratoconus or normal.
    Assumes 'keratoconus' is class 0 and 'normal' is class 1.
    """
    inputs = preprocess_image(image)
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()

    # Return the class based on the index (0 = KERATOCONUS, 1 = NORMAL)
    return 'KERATOCONUS' if predicted_class_idx == 0 else 'NORMAL'

# Flask route to handle image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Open the image file and run prediction
        image = Image.open(io.BytesIO(file.read()))
        prediction = predict_keratoconus(image)
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask route to serve the index file
@app.route("/")
def index():
    return send_file("index.html")

if __name__ == "__main__":
    app.run(debug=True)