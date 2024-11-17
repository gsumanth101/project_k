from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import cv2
import os
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

app = Flask(__name__)
app.secret_key = 'sumanthyadav1237'  # Secret key for flash messages

# Load Hugging Face Vision Transformer model and feature extractor
model_name = "google/vit-base-patch16-224-in21k"  # Pretrained ViT model
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded eye image
def preprocess_image(image_path):
    """  
    Preprocess the input image for the ViT model.
    """
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    inputs = feature_extractor(images=image, return_tensors="pt")  # Preprocess image
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU if available
    return inputs

# Function to predict if the image shows keratoconus
def predict_keratoconus(image_path):
    """
    Predict if the image is of keratoconus or normal.
    Assumes 'keratoconus' is class 0 and 'normal' is class 1.
    """
    inputs = preprocess_image(image_path)
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()

    # Return the class based on the index (0 = KERATOCONUS, 1 = NORMAL)
    return 'KERATOCONUS' if predicted_class_idx == 0 else 'NORMAL'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', filename)
            file.save(file_path)
            result = predict_keratoconus(file_path)

            if result == 'INVALID_IMAGE':
                flash('Invalid image. Please upload a valid eye image.')
                return redirect(request.url)

            # Simulate 3D Eye Structure Visualization
            img_data = simulate_3d_eye_structure(result)

            return render_template('index.html', filename=filename, result=result, img_data=img_data)
        else:
            flash('Invalid file type. Only eye images allowed.')
            return redirect(request.url)
    return render_template('index.html')

# Function to create a simulated 3D structure visualization for the eye
def simulate_3d_eye_structure(prediction_result):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) + 0.2 * np.exp(-(X**2 + Y**2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('3D Eye Structure Progression')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Deformation')

    # Convert plot to PNG image and encode as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

if __name__ == '__main__':
    app.run(debug=True)