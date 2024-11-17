import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse, Circle
import cv2
import google.generativeai as genai

# Configure the Google Gemini API key
genai.configure(api_key='AIzaSyBIcL3yE_e-T3l88v-MEGF0nWXs3cPXvMc')

# Load the keratoconus model
model = load_model('63.h5')

# Preprocess the image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (48, 48))  # Resize to 48x48 for model input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = image / 255.0  # Normalize the pixel values
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

# Predict keratoconus based on the image
def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)

    if prediction[0][0] <= 0.5:
        return 'KERATOCONUS'
    else:
        return 'NON_KERATOCONUS'

# Simulate a 3D eye progression (replace with diffusion model for real-world use)
def generate_3d_eye_structure():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.sin(np.sqrt(X**2 + Y**2)) + 0.2 * np.exp(-(X**2 + Y**2))
    
    # 3D Plot of progression
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('3D Progression of Keratoconus')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Deformation')

# Create a cartoon-style eye visualization with pupil, iris, and deformation if keratoconus is predicted
def draw_cartoon_eye(prediction_result):
    fig, ax = plt.subplots()

    # Draw the white part of the eye (sclera)
    eye_white = Circle((0.5, 0.5), 0.4, color='white', edgecolor='black', lw=2)
    ax.add_patch(eye_white)
    
    # Draw the iris
    iris_color = 'blue' if prediction_result == 'NON_KERATOCONUS' else 'red'
    iris = Circle((0.5, 0.5), 0.2, color=iris_color)
    ax.add_patch(iris)
    
    # Draw the pupil
    pupil = Circle((0.5, 0.5), 0.07, color='black')
    ax.add_patch(pupil)
    
    # If keratoconus is detected, show a deformation in the iris
    if prediction_result == 'KERATOCONUS':
        # Deform the iris to simulate the keratoconus effect
        deformed_iris = Ellipse((0.55, 0.45), 0.25, 0.15, color=iris_color, alpha=0.7)
        ax.add_patch(deformed_iris)

    # Formatting the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Eye with {prediction_result}')

    plt.show()

# Main logic
if __name__ == "__main__":
    # Load an example eye image (you can provide the real eye image path)
    image_path = r'data\Independent Test Set\Normal\case2\NOR_2_CT_A.jpg'

    # Predict if it's keratoconus or not
    result = predict_image(image_path)
    print(f'The image is a {result}')

    # Visualize cartoon-style eye with prediction
    draw_cartoon_eye(result)
    
    # Simulate 3D progression if keratoconus is detected
    if result == 'KERATOCONUS':
        generate_3d_eye_structure()
        
        # Fetch additional information from the Gemini API
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Write about keratoconus progression")
        print(f"Additional information about keratoconus: {response.text}")