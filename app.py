import google.generativeai as genai
from tensorflow.keras.models import load_model
import numpy as np

# Configure the Google Gemini API key
genai.configure(api_key='AIzaSyBIcL3yE_e-T3l88v-MEGF0nWXs3cPXvMc')

# Load the keratoconus model
model = load_model('63.h5')

# Preprocess the image for prediction
def preprocess_image(image_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img = image.img_to_array(img)
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)

    if prediction[0][0] <= 0.5:
        return 'KERATOCONUS'
    else:
        return 'NON_KERATOCONUS'

# Get text from the Gemini API about keratoconus
def get_gemini_response():
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Write about keratoconus")
    return response.text

# Main logic
image_path = r'data\Independent Test Set\Keratoconus\case1\KCN_1_CT_A.jpg'
result = predict_image(image_path)

# Print prediction
print(f'The image is a {result}')

# If it's keratoconus, fetch additional info from the Gemini API
if result == 'KERATOCONUS':
    gemini_response = get_gemini_response()
    print(f"Additional information about keratoconus: {gemini_response}")