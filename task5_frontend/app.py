"""
Task 5: Application Interface (Frontend)
Professional Flask web interface for face mask detection
"""

from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'task5_frontend/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/custom_cnn_best.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("✓ Model loaded successfully")

# Class names
CLASS_NAMES = ['With Mask', 'Without Mask', 'Incorrect Mask']
CLASS_COLORS = {
    'With Mask': '#2ecc71',
    'Without Mask': '#e74c3c',
    'Incorrect Mask': '#f39c12'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    
    Args:
        image_path: Path to image file
        target_size: Target size for model input
        
    Returns:
        Preprocessed image array
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch, image

def predict_mask(image_path):
    """
    Predict face mask status
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    image_batch, original_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(image_batch, verbose=0)
    class_id = np.argmax(predictions[0])
    confidence = float(predictions[0][class_id])
    
    # Get all class probabilities
    class_probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i] * 100)
        for i in range(len(CLASS_NAMES))
    }
    
    return {
        'class': CLASS_NAMES[class_id],
        'confidence': confidence * 100,
        'class_probabilities': class_probabilities,
        'color': CLASS_COLORS[CLASS_NAMES[class_id]]
    }

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_mask(filepath)
        
        # Read image and convert to base64
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        result['image'] = f"data:image/jpeg;base64,{image_data}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("=" * 60)
    print("FACE MASK DETECTION - WEB APPLICATION")
    print("=" * 60)
    print("\n✓ Server starting...")
    print("✓ Access the application at: http://localhost:5000")
    print("\n" + "=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
