import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image

# Load Vision Transformer (ViT) Model
vit_model_name = "google/vit-base-patch16-224-in21k"
vit_model = ViTForImageClassification.from_pretrained(vit_model_name)
vit_feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)

# Load Language Model for Generating Information
lm_model_name = "gpt2"  # Update to "gpt2" or a lighter model if necessary
lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name, use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(lm_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)
lm_model.to(device)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = vit_feature_extractor(images=image, return_tensors="pt").to(device)
    return inputs

def predict_keratoconus(image_path):
    inputs = preprocess_image(image_path)
    with torch.no_grad():
        outputs = vit_model(**inputs)
        predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()
    return 'KERATOCONUS' if predicted_class_idx == 0 else 'NORMAL'

def generate_keratoconus_info(stage):
    prompt = f"Keratoconus stage: {stage}. Provide details, remedies, and specialists."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = lm_model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
