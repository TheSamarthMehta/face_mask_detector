# 📋 Project Execution Guide

Step-by-step guide to execute all tasks in the Face Mask Detection project.

## ⚙️ Prerequisites Setup

Before starting, ensure you have:

```bash
# 1. Python 3.9+ installed
python --version

# 2. Virtual environment created and activated
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Dependencies installed
pip install -r requirements.txt

# 4. Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

---

## 📝 Task Execution Timeline

### ✅ Task 1: Problem Definition & Dataset Acquisition (19/01 - 24/01)

**Objective**: Analyze dataset and define the problem

**Steps:**

```bash
# Navigate to project root
cd face_mask_detection

# Run dataset analysis
python task1_problem_definition/analyze_dataset.py
```

**Expected Output:**

- Total images and annotations count
- Class distribution (with_mask, without_mask, mask_weared_incorrect)
- Data quality assessment
- Class imbalance ratio
- Visualization: `task1_problem_definition/class_distribution.png`

**What to verify:**

- ✓ Dataset contains 853 images
- ✓ Three classes are present
- ✓ Class distribution chart is generated
- ✓ Imbalance ratio is calculated

---

### ✅ Task 2: Data Pre-processing & Augmentation (26/01 - 31/01)

**Objective**: Preprocess images and apply augmentation

**Steps:**

```bash
# Run preprocessing pipeline
python task2_data_preprocessing/preprocess_data.py
```

**Expected Output:**

- Extracted and resized face regions (224x224)
- Normalized images (0-1 range)
- Augmented dataset (flipping, rotation, brightness)
- Train/Val/Test split (72%/8%/20%)
- Files in `task2_data_preprocessing/processed_data/`:
  - `X_train.npy`, `y_train.npy`
  - `X_val.npy`, `y_val.npy`
  - `X_test.npy`, `y_test.npy`
  - `metadata.json`

**What to verify:**

- ✓ Training samples > 5000
- ✓ All .npy files are created
- ✓ Sample images visualization is saved
- ✓ Class distribution is maintained after split

**Time Required**: 5-10 minutes

---

### ✅ Task 3: Model Architecture Design & Training (02/02 - 07/02)

**Objective**: Design and train custom CNN

**Steps:**

```bash
# Train custom CNN model
python task3_model_training/train_model.py
```

**Architecture:**

- Input: 224x224x3
- Conv Block 1: 32 filters
- Conv Block 2: 64 filters
- Conv Block 3: 128 filters
- Conv Block 4: 256 filters
- Dense: 512 → 256 → 3 (output)
- Regularization: Batch Norm, Dropout

**Expected Output:**

- Model training progress (epochs)
- Training and validation metrics
- Best model: `models/custom_cnn_best.h5`
- Final model: `models/custom_cnn_final.h5`
- Training history: `task3_model_training/training_history.json`
- Visualization: `task3_model_training/training_history.png`

**What to verify:**

- ✓ Model trains without errors
- ✓ Validation accuracy improves
- ✓ Model files are saved
- ✓ Training curves show convergence
- ✓ No severe overfitting (train-val gap < 10%)

**Time Required**: 30-60 minutes (GPU), 2-4 hours (CPU)

**Hyperparameters:**

- Learning rate: 0.001
- Batch size: 32
- Epochs: 50 (with early stopping)
- Optimizer: Adam

---

### ✅ Task 4: Evaluation & Hyperparameter Tuning (09/02 - 14/02)

**Objective**: Evaluate model and detect overfitting

**Steps:**

```bash
# Evaluate trained model
python task4_evaluation/evaluate_model.py
```

**Expected Output:**

- Test accuracy and loss
- Classification report (precision, recall, F1)
- Confusion matrix: `task4_evaluation/confusion_matrix.png`
- Per-class metrics: `task4_evaluation/class_metrics.png`
- Training analysis: `task4_evaluation/training_analysis.png`
- Results: `task4_evaluation/evaluation_results.json`

**What to verify:**

- ✓ Test accuracy > 90%
- ✓ Confusion matrix shows good predictions
- ✓ No class is severely underperforming
- ✓ Overfitting analysis is reasonable

**Time Required**: 5-10 minutes

**Evaluation Metrics:**

- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Train-validation gap analysis

---

### ✅ Task 5: Application Interface - Frontend (16/02 - 21/02)

**Objective**: Create web interface with Flask

**Steps:**

```bash
# Start Flask web server
python task5_frontend/app.py
```

**Access:**

- URL: http://localhost:5000
- Upload images through web interface
- Get real-time predictions

**Features:**

- ✓ Modern, professional UI with gradient design
- ✓ Drag-and-drop image upload
- ✓ Real-time prediction
- ✓ Confidence scores
- ✓ Color-coded results
- ✓ Probability bars for all classes
- ✓ Responsive design

**What to verify:**

- ✓ Server starts without errors
- ✓ Web page loads correctly
- ✓ Image upload works
- ✓ Predictions are accurate
- ✓ UI is professional and responsive

**Time Required**: Instant (server startup)

**Test Images:**

- Try with mask images → Should predict "With Mask"
- Try without mask images → Should predict "Without Mask"
- Try various lighting and angles

---

### ✅ Task 6: Advanced Optimization (23/02 - 28/02)

**Objective**: Implement transfer learning and compare models

**Steps:**

```bash
# Train MobileNetV2 model
python task6_advanced_optimization/transfer_learning.py
```

**Transfer Learning Model:**

- Base: MobileNetV2 (pre-trained on ImageNet)
- Custom classifier on top
- Fine-tuning (optional)

**Expected Output:**

- MobileNetV2 model: `models/mobilenetv2_best.h5`
- Training history: `task6_advanced_optimization/mobilenetv2_history.json`
- Model comparison: `task6_advanced_optimization/model_comparison.png`
- Comparison results: `task6_advanced_optimization/comparison_results.json`
- Best model: `models/best_model.h5`

**What to verify:**

- ✓ Transfer learning model trains successfully
- ✓ Comparison shows metrics for both models
- ✓ Best model is selected automatically
- ✓ Accuracy improvement is documented

**Time Required**: 15-30 minutes

**Expected Improvement:**

- 2-5% accuracy increase over custom CNN
- Faster convergence
- Better generalization

---

### ✅ Task 7: Backend Integration & Deployment (02/03 - 07/03)

**Objective**: Create production API and deployment configs

**Steps:**

```bash
# Option 1: Run FastAPI backend
python task7_deployment/api.py

# Option 2: Docker deployment
cd task7_deployment
docker-compose up --build
```

**FastAPI Endpoints:**

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Image prediction
- `GET /classes` - Available classes
- `GET /model-info` - Model details
- `GET /docs` - Interactive API documentation

**Access:**

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

**Deployment Configurations:**

- ✓ Dockerfile
- ✓ docker-compose.yml
- ✓ railway.toml (Railway)
- ✓ vercel.json (Vercel)

**What to verify:**

- ✓ API server starts successfully
- ✓ All endpoints respond correctly
- ✓ POST /predict accepts images
- ✓ Response format is correct
- ✓ Interactive docs are accessible

**Time Required**: Instant (server startup), 5-10 minutes (Docker build)

**Test API:**

```bash
# Health check
curl http://localhost:8000/health

# Predict (replace with actual image path)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

---

## 🎯 Complete Execution Checklist

### Before Starting

- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] Dataset available in `archive/` folder

### Task Execution

- [ ] Task 1: Dataset analysis completed
- [ ] Task 2: Data preprocessing completed
- [ ] Task 3: Custom CNN trained
- [ ] Task 4: Model evaluated
- [ ] Task 5: Flask app running
- [ ] Task 6: Transfer learning completed
- [ ] Task 7: API backend deployed

### Verification

- [ ] All visualizations generated
- [ ] Models saved in `models/` folder
- [ ] Test accuracy > 90%
- [ ] Web interface works correctly
- [ ] API responds to requests
- [ ] Documentation complete

---

## 📊 Expected Results Summary

| Task | Output            | Success Criteria                 |
| ---- | ----------------- | -------------------------------- |
| 1    | Dataset analysis  | 853 images, 3 classes identified |
| 2    | Processed data    | ~5000+ training samples          |
| 3    | Trained CNN       | Accuracy > 90%, model saved      |
| 4    | Evaluation        | Confusion matrix, metrics        |
| 5    | Web Interface     | Functional, professional UI      |
| 6    | Transfer Learning | Improved accuracy                |
| 7    | API Backend       | All endpoints working            |

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Error:**

```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**2. Out of Memory:**

```bash
# Solution: Reduce batch size in training scripts
# Edit train_model.py: batch_size=16 instead of 32
```

**3. Model Not Found:**

```bash
# Solution: Ensure Task 3 completed successfully
ls models/
# Should show custom_cnn_best.h5
```

**4. Port Already in Use:**

```bash
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :5000
kill -9 <PID>
```

---

## 🚀 Deployment to Production

### Railway

```bash
railway login
railway init
railway link
railway up
```

### Render

1. Push code to GitHub
2. Connect repository on Render
3. Set start command: `python task7_deployment/api.py`
4. Deploy

### Hugging Face Spaces

1. Create new Space
2. Upload files
3. Use provided Dockerfile
4. Deploy

---

## 📈 Performance Benchmarks

### Training Time (approximate)

- **Task 2**: 5-10 minutes
- **Task 3**: 30-60 minutes (GPU), 2-4 hours (CPU)
- **Task 6**: 15-30 minutes (GPU)

### Expected Accuracy

- **Custom CNN**: 92-95%
- **MobileNetV2**: 94-97%

### Inference Time

- **Single Image**: 50-100ms (GPU), 200-500ms (CPU)

---

## ✅ Project Completion

Once all tasks are complete:

1. ✓ Review all generated visualizations
2. ✓ Test the web interface thoroughly
3. ✓ Verify API endpoints
4. ✓ Check deployment configurations
5. ✓ Document any customizations
6. ✓ Prepare final presentation

---

## 📝 Final Deliverables

- [ ] Complete codebase
- [ ] Trained models
- [ ] Evaluation reports
- [ ] Working web interface
- [ ] API backend
- [ ] Deployment configs
- [ ] Documentation
- [ ] Project presentation

---

**Total Estimated Time**: 4-6 hours (with GPU), 8-10 hours (CPU only)

**Status**: Ready for execution and deployment! 🎉
