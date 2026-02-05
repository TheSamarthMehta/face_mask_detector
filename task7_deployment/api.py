"""
Task 7: Backend Integration & Deployment
FastAPI backend for face mask detection with deployment configuration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Mask Detection API",
    description="AI-powered face mask detection system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
CLASS_NAMES = ['With Mask', 'Without Mask', 'Incorrect Mask']
CLASS_COLORS = {
    'With Mask': '#2ecc71',
    'Without Mask': '#e74c3c',
    'Incorrect Mask': '#f39c12'
}

def load_model():
    """Load the trained model"""
    global MODEL
    try:
        model_path = os.environ.get('MODEL_PATH', 'models/best_model.h5')
        if not os.path.exists(model_path):
            model_path = 'models/custom_cnn_best.h5'
        
        MODEL = keras.models.load_model(model_path)
        logger.info(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()
    logger.info("✓ API server started")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess uploaded image
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed image array
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid image file")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Mask Detection API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Predict face mask status from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with class and confidence
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        image_batch = preprocess_image(image_bytes)
        
        # Predict
        predictions = MODEL.predict(image_batch, verbose=0)
        class_id = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_id])
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i] * 100)
            for i in range(len(CLASS_NAMES))
        }
        
        # Prepare response
        result = {
            "success": True,
            "prediction": {
                "class": CLASS_NAMES[class_id],
                "class_id": class_id,
                "confidence": confidence * 100,
                "color": CLASS_COLORS[CLASS_NAMES[class_id]]
            },
            "probabilities": class_probabilities
        }
        
        logger.info(f"Prediction: {CLASS_NAMES[class_id]} ({confidence*100:.2f}%)")
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get available classes"""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "input_shape": MODEL.input_shape,
        "output_shape": MODEL.output_shape,
        "classes": CLASS_NAMES
    }

if __name__ == "__main__":
    print("=" * 60)
    print("FACE MASK DETECTION - FASTAPI BACKEND")
    print("=" * 60)
    print("\n✓ Starting server...")
    print("✓ API Documentation: http://localhost:8000/docs")
    print("✓ Health Check: http://localhost:8000/health")
    print("\n" + "=" * 60)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
