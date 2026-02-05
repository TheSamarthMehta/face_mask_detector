# Task 5: Application Interface (Frontend)

## Professional Flask Web Application

A clean, modern, and user-friendly web interface for face mask detection.

## Features

### User Interface

- **Modern Design**: Professional gradient-based design with smooth animations
- **Drag & Drop**: Easy file upload with drag-and-drop support
- **Real-time Preview**: Instant image preview before analysis
- **Interactive Results**: Animated progress bars and color-coded predictions
- **Responsive**: Works on desktop and mobile devices

### Functionality

- **Image Upload**: Support for PNG, JPG, JPEG formats (max 16MB)
- **Preprocessing Pipeline**: Automatic image resizing and normalization
- **Real-time Prediction**: Fast inference using trained model
- **Confidence Scores**: Displays prediction confidence and all class probabilities
- **Error Handling**: User-friendly error messages
- **Health Check**: API endpoint for monitoring

## API Endpoints

### 1. Main Page

```
GET /
```

Renders the web interface

### 2. Prediction

```
POST /predict
```

**Request**: Multipart form-data with image file  
**Response**: JSON with prediction results

```json
{
  "class": "With Mask",
  "confidence": 95.67,
  "class_probabilities": {
    "With Mask": 95.67,
    "Without Mask": 2.15,
    "Incorrect Mask": 2.18
  },
  "color": "#2ecc71",
  "image": "data:image/jpeg;base64,..."
}
```

### 3. Health Check

```
GET /health
```

**Response**: Server and model status

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Color Coding

- 🟢 **Green** (#2ecc71): With Mask
- 🔴 **Red** (#e74c3c): Without Mask
- 🟠 **Orange** (#f39c12): Incorrect Mask

## File Structure

```
task5_frontend/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Web interface
├── uploads/            # Temporary upload storage
└── README.md          # Documentation
```

## Usage

### Start the server:

```bash
python task5_frontend/app.py
```

### Access the application:

```
http://localhost:5000
```

## Dependencies

- Flask
- TensorFlow/Keras
- OpenCV
- NumPy
- Werkzeug

## Security Features

- File type validation
- File size limit (16MB)
- Secure filename handling
- Error handling and validation

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge
