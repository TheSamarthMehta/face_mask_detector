# 🚀 Quick Start Guide

Get the Face Mask Detection System up and running in minutes!

## Prerequisites

- Python 3.9+
- pip
- 8GB+ RAM (for training)
- (Optional) NVIDIA GPU with CUDA for faster training

## Installation Steps

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd face_mask_detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Option A: Use Pre-trained Model (Fastest)

If you have a pre-trained model file (`models/best_model.h5`):

```bash
# Run Flask Web Interface
python task5_frontend/app.py

# Open browser and go to:
# http://localhost:5000
```

### 3. Option B: Train from Scratch

If starting from raw data:

```bash
# Step 1: Analyze dataset
python task1_problem_definition/analyze_dataset.py

# Step 2: Preprocess data (takes 5-10 minutes)
python task2_data_preprocessing/preprocess_data.py

# Step 3: Train custom CNN (takes 30-60 minutes)
python task3_model_training/train_model.py

# Step 4: Evaluate model
python task4_evaluation/evaluate_model.py

# Step 5: (Optional) Train transfer learning model
python task6_advanced_optimization/transfer_learning.py

# Step 6: Run web interface
python task5_frontend/app.py
```

## Testing the Application

### Web Interface (Flask)

1. Start Flask server:

```bash
python task5_frontend/app.py
```

2. Open browser: `http://localhost:5000`

3. Upload an image and see the prediction!

### API Backend (FastAPI)

1. Start FastAPI server:

```bash
python task7_deployment/api.py
```

2. Open API docs: `http://localhost:8000/docs`

3. Test the `/predict` endpoint with an image

### Command Line Test

```bash
# Test API with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

## Docker Deployment

```bash
# Build and run with docker-compose
cd task7_deployment
docker-compose up --build

# Access:
# Frontend: http://localhost:5000
# API: http://localhost:8000
```

## Common Issues

### Issue: Module not found

**Solution**: Ensure virtual environment is activated and all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Issue: Out of memory during training

**Solution**: Reduce batch size in training scripts:

- Edit `task3_model_training/train_model.py`
- Change `batch_size=32` to `batch_size=16` or `batch_size=8`

### Issue: Model file not found

**Solution**: Train the model first or ensure model file exists:

```bash
# Check if model exists
ls models/

# If not, train the model
python task3_model_training/train_model.py
```

### Issue: Port already in use

**Solution**: Change port or kill existing process:

```bash
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :5000
kill -9 <PID>
```

## Project Structure Overview

```
face_mask_detection/
├── task1_problem_definition/    # Dataset analysis
├── task2_data_preprocessing/    # Data preparation
├── task3_model_training/        # Custom CNN training
├── task4_evaluation/            # Model evaluation
├── task5_frontend/              # Flask web interface
├── task6_advanced_optimization/ # Transfer learning
├── task7_deployment/            # Production deployment
├── models/                      # Trained models
└── archive/                     # Dataset (images + annotations)
```

## Next Steps

After setup:

1. ✅ Explore the web interface
2. ✅ Test different images
3. ✅ Review evaluation metrics
4. ✅ Compare model performance
5. ✅ Deploy to production (Railway/Render/HuggingFace)

## Need Help?

- 📖 Read the main [README.md](README.md)
- 📖 Check task-specific README files in each folder
- 🐛 Open an issue on GitHub
- 📧 Contact support

## Tips for Best Results

1. **Dataset**: Ensure images are clear and faces are visible
2. **Training**: Use GPU if available for faster training
3. **Testing**: Test with various lighting conditions and angles
4. **Deployment**: Use Docker for consistent environments

---

**Estimated Time to Complete:**

- Quick Start (pre-trained): 5 minutes
- Full Training Pipeline: 1-2 hours
- Deployment Setup: 15-30 minutes

Happy Coding! 🎉
