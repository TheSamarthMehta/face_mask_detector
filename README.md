# 🎭 Face Mask Detection System

A complete end-to-end deep learning project for detecting face masks in images using custom CNN and transfer learning (MobileNetV2). This project includes data preprocessing, model training, evaluation, a professional web interface, and deployment-ready configurations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-teal.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Task Breakdown](#task-breakdown)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a complete face mask detection system as part of a Deep Learning course project. The system can classify images into three categories:

- ✅ **With Mask**: Person wearing a mask correctly
- ❌ **Without Mask**: Person not wearing a mask
- ⚠️ **Incorrect Mask**: Person wearing a mask incorrectly

### Problem Statement

Develop a deep learning system to detect and classify face masks in images for COVID-19 safety compliance monitoring.

### Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Web Frameworks**: Flask, FastAPI
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Docker, Railway, Render

---

## ✨ Features

### 🔬 Machine Learning

- Custom CNN architecture from scratch
- Transfer learning with MobileNetV2
- Comprehensive data augmentation
- Hyperparameter tuning
- Overfitting/underfitting detection
- Model comparison and selection

### 🎨 User Interface

- Professional, modern web design
- Drag-and-drop image upload
- Real-time prediction
- Confidence scores and probability bars
- Color-coded results
- Responsive design

### 🚀 Backend & API

- RESTful API with FastAPI
- Automatic API documentation (Swagger)
- Health check endpoints
- CORS support
- Error handling and validation
- Production-ready deployment configs

### 📊 Evaluation & Visualization

- Confusion matrix
- Classification reports
- Training history plots
- Per-class metrics
- Model comparison charts

---

## 📁 Project Structure

```
face_mask_detection/
├── archive/                          # Dataset
│   ├── images/                      # Image files (853 images)
│   └── annotations/                 # XML annotations (PASCAL VOC format)
│
├── task1_problem_definition/         # Task 1
│   ├── analyze_dataset.py           # Dataset analysis script
│   ├── class_distribution.png       # Visualization
│   └── README.md
│
├── task2_data_preprocessing/         # Task 2
│   ├── preprocess_data.py           # Preprocessing pipeline
│   ├── processed_data/              # Processed datasets
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_val.npy
│   │   ├── y_val.npy
│   │   ├── X_test.npy
│   │   ├── y_test.npy
│   │   └── metadata.json
│   ├── sample_images.png
│   └── README.md
│
├── task3_model_training/             # Task 3
│   ├── train_model.py               # Custom CNN training
│   ├── training_history.png
│   ├── training_history.json
│   └── README.md
│
├── task4_evaluation/                 # Task 4
│   ├── evaluate_model.py            # Model evaluation
│   ├── confusion_matrix.png
│   ├── class_metrics.png
│   ├── training_analysis.png
│   ├── evaluation_results.json
│   └── README.md
│
├── task5_frontend/                   # Task 5
│   ├── app.py                       # Flask web application
│   ├── templates/
│   │   └── index.html               # Professional UI
│   ├── uploads/                     # Temporary upload storage
│   └── README.md
│
├── task6_advanced_optimization/      # Task 6
│   ├── transfer_learning.py         # MobileNetV2 implementation
│   ├── mobilenetv2_history.json
│   ├── model_comparison.png
│   ├── comparison_results.json
│   └── README.md
│
├── task7_deployment/                 # Task 7
│   ├── api.py                       # FastAPI backend
│   ├── Dockerfile                   # Docker configuration
│   ├── docker-compose.yml           # Multi-container setup
│   ├── railway.toml                 # Railway config
│   ├── vercel.json                  # Vercel config
│   └── README.md
│
├── models/                           # Trained models
│   ├── custom_cnn_best.h5          # Best custom CNN
│   ├── custom_cnn_final.h5         # Final custom CNN
│   ├── mobilenetv2_best.h5         # Best transfer learning model
│   └── best_model.h5               # Selected best model for deployment
│
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA for GPU acceleration

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd face_mask_detection
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 🚀 Usage

### Complete Pipeline (All Tasks)

Run each task sequentially:

```bash
# Task 1: Analyze dataset
python task1_problem_definition/analyze_dataset.py

# Task 2: Preprocess data
python task2_data_preprocessing/preprocess_data.py

# Task 3: Train custom CNN
python task3_model_training/train_model.py

# Task 4: Evaluate model
python task4_evaluation/evaluate_model.py

# Task 5: Run Flask web app
python task5_frontend/app.py

# Task 6: Transfer learning
python task6_advanced_optimization/transfer_learning.py

# Task 7: Run FastAPI backend
python task7_deployment/api.py
```

### Quick Start (Using Pre-trained Model)

If you have a pre-trained model:

```bash
# Run Flask frontend
python task5_frontend/app.py
# Access at http://localhost:5000

# Or run FastAPI backend
python task7_deployment/api.py
# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

---

## 📝 Task Breakdown

### Task 1: Problem Definition & Dataset Acquisition

- **Objective**: Define the problem and analyze the dataset
- **Output**: Class distribution analysis, data quality assessment
- **Execution**: 19/01 - 24/01
- **Files**: `task1_problem_definition/`

### Task 2: Data Pre-processing & Augmentation

- **Objective**: Preprocess images and apply augmentation
- **Techniques**: Resize, normalize, rotation, flipping, brightness adjustment
- **Dataset Split**: 72% train, 8% validation, 20% test
- **Execution**: 26/01 - 31/01
- **Files**: `task2_data_preprocessing/`

### Task 3: Model Architecture Design & Training

- **Objective**: Design and train custom CNN
- **Architecture**: 4 convolutional blocks + dense layers
- **Regularization**: Dropout, batch normalization, early stopping
- **Optimizer**: Adam with learning rate scheduling
- **Execution**: 02/02 - 07/02
- **Files**: `task3_model_training/`

### Task 4: Evaluation & Hyperparameter Tuning

- **Objective**: Evaluate model and detect overfitting
- **Metrics**: Accuracy, loss, precision, recall, F1-score
- **Visualization**: Confusion matrix, training curves
- **Execution**: 09/02 - 14/02
- **Files**: `task4_evaluation/`

### Task 5: Application Interface (Frontend)

- **Objective**: Create professional web interface
- **Framework**: Flask
- **Features**: Image upload, real-time prediction, visualization
- **Execution**: 16/02 - 21/02
- **Files**: `task5_frontend/`

### Task 6: Advanced Optimization

- **Objective**: Implement transfer learning
- **Model**: MobileNetV2 (pre-trained on ImageNet)
- **Comparison**: Custom CNN vs Transfer Learning
- **Execution**: 23/02 - 28/02
- **Files**: `task6_advanced_optimization/`

### Task 7: Backend Integration & Deployment

- **Objective**: Production deployment setup
- **Framework**: FastAPI
- **Deployment**: Docker, Railway, Render, Hugging Face Spaces
- **Execution**: 02/03 - 07/03
- **Files**: `task7_deployment/`

---

## 📊 Model Performance

### Custom CNN

- **Architecture**: 4 Conv blocks + 2 Dense layers
- **Parameters**: ~5M
- **Training Time**: ~30 minutes (GPU)
- **Test Accuracy**: ~92-95%
- **Test Loss**: ~0.20-0.25

### MobileNetV2 (Transfer Learning)

- **Architecture**: MobileNetV2 + Custom classifier
- **Parameters**: ~3.5M (2.5M frozen)
- **Training Time**: ~15 minutes (GPU)
- **Test Accuracy**: ~94-97%
- **Test Loss**: ~0.15-0.20

### Best Model Selection

The best performing model (highest test accuracy) is automatically saved as `models/best_model.h5` for deployment.

---

## 🚢 Deployment

### Local Deployment

```bash
# Flask frontend
python task5_frontend/app.py

# FastAPI backend
python task7_deployment/api.py
```

### Docker Deployment

```bash
# Build and run
docker-compose -f task7_deployment/docker-compose.yml up --build

# Access services
# Frontend: http://localhost:5000
# API: http://localhost:8000
```

### Cloud Deployment

#### Railway

```bash
railway login
railway init
railway up
```

#### Render

1. Connect GitHub repository
2. Create new Web Service
3. Set start command: `python task7_deployment/api.py`

#### Hugging Face Spaces

1. Create new Space
2. Upload files and model
3. Set SDK to Docker
4. Use provided Dockerfile

---

## 📚 API Documentation

### Endpoints

#### POST /predict

Upload image for prediction

**Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

**Response:**

```json
{
  "success": true,
  "prediction": {
    "class": "With Mask",
    "confidence": 95.67,
    "color": "#2ecc71"
  },
  "probabilities": {
    "With Mask": 95.67,
    "Without Mask": 2.15,
    "Incorrect Mask": 2.18
  }
}
```

#### GET /health

Health check

#### GET /classes

Get available classes

#### GET /model-info

Get model information

### Interactive Documentation

Access Swagger UI at: `http://localhost:8000/docs`

---

## 🖼️ Screenshots

### Web Interface

- Modern, professional design with gradient background
- Drag-and-drop file upload
- Real-time prediction with confidence scores
- Color-coded results (Green/Red/Orange)
- Animated probability bars

### Visualizations

- Class distribution charts
- Training history plots
- Confusion matrix
- Model comparison graphs
- Per-class performance metrics

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Student Name** - Darshan University
- **Course**: Machine Learning & Deep Learning
- **Semester**: 2024

---

## 🙏 Acknowledgments

- Dataset: Face Mask Detection Dataset
- Pre-trained Model: MobileNetV2 (ImageNet)
- Frameworks: TensorFlow, Keras, Flask, FastAPI
- University: Darshan University

---

## 📞 Support

For issues or questions:

- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 🔄 Project Status

✅ Task 1: Problem Definition - **Complete**  
✅ Task 2: Data Preprocessing - **Complete**  
✅ Task 3: Model Training - **Complete**  
✅ Task 4: Evaluation - **Complete**  
✅ Task 5: Frontend Interface - **Complete**  
✅ Task 6: Advanced Optimization - **Complete**  
✅ Task 7: Deployment - **Complete**

**Status**: 🎉 **Production Ready**

---

## 📈 Future Enhancements

- [ ] Real-time video detection
- [ ] Mobile application (iOS/Android)
- [ ] Multi-face detection in single image
- [ ] Model quantization for edge devices
- [ ] Additional augmentation techniques
- [ ] Ensemble models
- [ ] A/B testing framework

---

## 📝 Notes

- All code is well-commented and easy to understand
- Each task has its own README with detailed instructions
- Models are saved in H5 format for easy deployment
- Professional UI with modern design principles
- Production-ready deployment configurations included

---

**Made with ❤️ for Deep Learning Course Project**
