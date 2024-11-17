import os
import google.generativeai as genai
from flask import Flask, render_template, request, url_for, redirect, flash, abort
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import matplotlib.pyplot as plt
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SECRET_KEY'] = 'sumanth'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://keratoconus_owner:qvmlDYL5uPp1@ep-lucky-cherry-a8rvw356.eastus2.azure.neon.tech/keratoconus?sslmode=require'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(60), nullable=False)

class Report(db.Model):
    __tablename__ = 'reports'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    stage_info = db.Column(db.Text, nullable=False)
    graph_path = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize Hugging Face token directly in code (if needed for other models)
os.environ["HUGGINGFACE_TOKEN"] = "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"  # Replace with your actual token
genai.configure(api_key='AIzaSyBIcL3yE_e-T3l88v-MEGF0nWXs3cPXvMc')

# Load Vision Transformer (ViT) model for image classification
vit_model_name = "google/vit-base-patch16-224-in21k"
vit_model = ViTForImageClassification.from_pretrained(vit_model_name)
vit_feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)

# Image preprocessing for ViT
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = vit_feature_extractor(images=image, return_tensors="pt").to(device)
    return inputs

# Function to predict keratoconus from eye image
def predict_keratoconus(image_path):
    inputs = preprocess_image(image_path)
    with torch.no_grad():
        outputs = vit_model(**inputs)
        predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()
    return 'KERATOCONUS' if predicted_class_idx == 0 else 'NORMAL'

# Function to generate detailed information about keratoconus using Gemini API (via genai)
def generate_keratoconus_info(stage):
    prompt = f"""
    Please provide a detailed report on Keratoconus for the patient at stage: {stage}. 
    Your report should include the following sections:

    1. Condition Overview: 
       - A comprehensive description of what Keratoconus is and how it affects the eye.
    
    2. Symptoms: 
       - The common symptoms associated with this stage of Keratoconus.

    3. Diagnosis: 
       - How this stage is diagnosed and any tests or procedures that are typically used.

    4. Possible Remedies:
       - A list of potential treatments or interventions for this stage, including medical and surgical options.
    
    5. Expected Progression:
       - A prediction of how the condition will progress in the coming years if left untreated, along with any factors that may influence this progression.
    
    6. Visualizations:
       - Provide a graph or chart that shows the expected progression of Keratoconus from the early stage to the advanced stage, including data points or trends.

    7. Nearby Specialists and Doctors:
       - Information about specialized doctors or clinics that treat this condition, including locations and contact information.

    8. Patient and Family Support:
       - Advice on how the patient’s family, friends, and medical professionals can support the patient during this stage.

    9. Additional Recommendations:
       - Any other information, lifestyle recommendations, or advice to improve quality of life or manage the condition.

    Please ensure that the information is well-structured, includes graphs/charts for visual representation, and is easy to understand.
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    response_text = response.text

    # Split the response into individual sections
    sections = response_text.split('\n')
    sections = [section.strip() for section in sections if section.strip()]
    return sections

# Function to generate a graph representing progression
def generate_progression_graph(stage):
    # Define progression data (this can be adjusted based on real-world progression data)
    stages = ['Early', 'Moderate', 'Advanced']
    years = [0, 1, 2, 3, 4, 5]  # years of progression
    progression = {
        'Early': [0, 10, 20, 25, 30, 35],    # Hypothetical data for Early stage
        'Moderate': [0, 20, 40, 50, 60, 70],  # Hypothetical data for Moderate stage
        'Advanced': [0, 30, 50, 70, 85, 100]  # Hypothetical data for Advanced stage
    }

    # Select the correct progression data based on the stage
    progression_data = progression.get(stage, progression['Early'])

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(years, progression_data, marker='o', linestyle='-', color='b')

    # Add titles and labels
    plt.title(f"Progression of Keratoconus: {stage} Stage", fontsize=14)
    plt.xlabel("Years", fontsize=12)
    plt.ylabel("Condition Severity (%)", fontsize=12)
    plt.grid(True)

    # Save the graph as an image
    graph_path = f"static/progression_{stage}.png"
    plt.savefig(graph_path)
    plt.close()

    return graph_path

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload_image.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash("No file selected!")
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        flash("No image selected!")
        return redirect(url_for('home'))

    # Save uploaded image
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    # Make prediction
    result = predict_keratoconus(image_path)

    # Generate information based on the predicted stage
    stage_info = ""
    graph_path = ""
    if result == 'KERATOCONUS':
        stage_info = generate_keratoconus_info('Advanced')
        graph_path = generate_progression_graph('Advanced')
    else:
        stage_info = generate_keratoconus_info('Early')
        graph_path = generate_progression_graph('Early')

    # Prepare data for display
    return render_template('result.html', result=result, stage_info=stage_info, image_path=image_path, graph_path=graph_path)

# Route for login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=request.form)

# Route for report details
@app.route('/report/<int:report_id>')
@login_required
def report_details(report_id):
    report = Report.query.get_or_404(report_id)
    if report.user_id != current_user.id:
        abort(403)
    today = datetime.today()
    age = today.year - current_user.date_of_birth.year - ((today.month, today.day) < (current_user.date_of_birth.month, current_user.date_of_birth.day))
    return render_template('report_details.html', report=report, age=age)

# Route for dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.timestamp.desc()).all()
    today = datetime.today()
    age = today.year - current_user.date_of_birth.year - ((today.month, today.day) < (current_user.date_of_birth.month, current_user.date_of_birth.day))
    return render_template('dash.html', age=age, reports=reports)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/track_progression')
@login_required
def track_progression():
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.timestamp.desc()).all()
    return render_template('track_progression.html', reports=reports)

@app.route('/upload_image', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file selected!")
            return redirect(url_for('upload_image'))

        file = request.files['image']
        if file.filename == '':
            flash("No image selected!")
            return redirect(url_for('upload_image'))

        # Save uploaded image
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Make prediction
        result = predict_keratoconus(image_path)

        # Generate information based on the predicted stage
        stage_info = ""
        graph_path = ""
        if result == 'KERATOCONUS':
            stage_info = generate_keratoconus_info('Advanced')
            graph_path = generate_progression_graph('Advanced')
        else:
            stage_info = generate_keratoconus_info('Early')
            graph_path = generate_progression_graph('Early')

        # Save report to database
        report = Report(user_id=current_user.id, image_path=image_path, result=result, stage_info=stage_info, graph_path=graph_path)
        db.session.add(report)
        db.session.commit()

        # Prepare data for display
        return render_template('result.html', result=result, stage_info=stage_info, image_path=image_path, graph_path=graph_path)

    return render_template('upload_image.html')

# Route for logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        date_of_birth = request.form.get('date_of_birth')
        phone = request.form.get('phone')
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(name=name, email=email, date_of_birth=date_of_birth, phone=phone, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=request.form)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)