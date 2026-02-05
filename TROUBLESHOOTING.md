# 🔧 Troubleshooting Guide

Common issues and solutions for the Face Mask Detection project.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Training Issues](#training-issues)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [Deployment Issues](#deployment-issues)

---

## Installation Issues

### Issue: pip install fails with dependency conflicts

**Error:**

```
ERROR: Cannot install package X because these package versions have conflicting dependencies
```

**Solution:**

```bash
# Create fresh virtual environment
python -m venv venv_new
venv_new\Scripts\activate  # Windows
source venv_new/bin/activate  # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies one by one
pip install tensorflow==2.13.0
pip install opencv-python==4.8.0.76
pip install -r requirements.txt
```

### Issue: TensorFlow installation fails on Windows

**Error:**

```
ERROR: Could not find a version that satisfies the requirement tensorflow
```

**Solution:**

```bash
# Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/16/release/vc_redist.x64.exe

# Install TensorFlow for CPU
pip install tensorflow-cpu==2.13.0

# Or install TensorFlow for GPU (requires CUDA)
pip install tensorflow-gpu==2.13.0
```

### Issue: OpenCV import error

**Error:**

```python
ImportError: DLL load failed: The specified module could not be found.
```

**Solution:**

```bash
# Uninstall and reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.0.76

# If still fails, try headless version
pip install opencv-python-headless==4.8.0.76
```

---

## Training Issues

### Issue: Out of Memory (OOM) during training

**Error:**

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**

```python
# Option 1: Reduce batch size
# In train_model.py or transfer_learning.py, change:
batch_size = 16  # or even 8 instead of 32

# Option 2: Use mixed precision training
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Option 3: Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Issue: Training is very slow

**Problem:** Training takes hours on CPU

**Solution:**

```bash
# Check if GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU:
# 1. Reduce dataset size for testing
# 2. Reduce number of epochs
# 3. Use smaller batch size
# 4. Use pre-trained model (skip Task 3, use Task 6)

# In preprocess_data.py, limit data:
# Add this after loading annotations:
annotation_files = annotation_files[:100]  # Use only 100 images for testing
```

### Issue: Model not improving (accuracy stuck)

**Problem:** Validation accuracy plateaus early

**Solution:**

```python
# 1. Check learning rate
# Try different learning rates: 0.0001, 0.001, 0.01

# 2. Check data augmentation
# Ensure augmentation is working (view sample_images.png)

# 3. Check class imbalance
# Run task1 to verify balanced classes

# 4. Increase model capacity
# Add more layers or filters in train_model.py

# 5. Train longer
epochs = 100  # Instead of 50
```

### Issue: NaN loss during training

**Error:**

```
Epoch 5: loss: nan, accuracy: 0.0
```

**Solution:**

```python
# 1. Check for data issues
# Ensure images are normalized: values in [0, 1]

# 2. Reduce learning rate
learning_rate = 0.0001  # Instead of 0.001

# 3. Add gradient clipping
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

# 4. Check for extreme values
# In preprocess_data.py, verify normalization:
print(f"Min: {X_train.min()}, Max: {X_train.max()}")
# Should be: Min: 0.0, Max: 1.0
```

---

## Runtime Errors

### Issue: Model file not found

**Error:**

```
FileNotFoundError: models/custom_cnn_best.h5
```

**Solution:**

```bash
# 1. Ensure Task 3 completed successfully
python task3_model_training/train_model.py

# 2. Check if models directory exists
mkdir models  # Windows
# or
mkdir -p models  # Linux/Mac

# 3. Verify model was saved
ls models/  # or dir models\ on Windows

# 4. Use correct path in code
model_path = os.path.join('models', 'custom_cnn_best.h5')
```

### Issue: ImportError for local modules

**Error:**

```
ModuleNotFoundError: No module named 'task1_problem_definition'
```

**Solution:**

```bash
# Run scripts from project root
cd face_mask_detection
python task1_problem_definition/analyze_dataset.py

# Don't run from inside task folders
# DON'T DO: cd task1_problem_definition && python analyze_dataset.py
```

### Issue: XML parsing error

**Error:**

```
xml.etree.ElementTree.ParseError: syntax error
```

**Solution:**

```python
# Add error handling in analyze_dataset.py or preprocess_data.py
try:
    tree = ET.parse(xml_path)
    root = tree.getroot()
except ET.ParseError as e:
    print(f"Skipping corrupted XML: {xml_path}")
    continue
```

---

## Performance Issues

### Issue: Slow inference time

**Problem:** Prediction takes > 1 second per image

**Solution:**

```python
# 1. Batch predictions
images_batch = np.array([img1, img2, img3, ...])
predictions = model.predict(images_batch)

# 2. Use model.predict() instead of model()
# Faster: predictions = model.predict(X_test)
# Slower: predictions = model(X_test)

# 3. Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 4. Use GPU if available
with tf.device('/GPU:0'):
    predictions = model.predict(images)
```

### Issue: High memory usage

**Problem:** Application uses > 4GB RAM

**Solution:**

```python
# 1. Load model only once (not per request)
# In Flask/FastAPI, load model at startup

# 2. Clear session after prediction
import gc
from tensorflow.keras import backend as K

def predict_and_clear():
    result = model.predict(image)
    K.clear_session()
    gc.collect()
    return result

# 3. Use smaller model
# Use MobileNetV2 instead of Custom CNN

# 4. Quantize model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

---

## Deployment Issues

### Issue: Flask server won't start

**Error:**

```
Address already in use: Port 5000
```

**Solution:**

```bash
# Option 1: Kill existing process
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :5000
kill -9 <PID>

# Option 2: Use different port
# In app.py:
app.run(port=5001)

# Option 3: Set environment variable
$env:FLASK_RUN_PORT=5001  # Windows PowerShell
export FLASK_RUN_PORT=5001  # Linux/Mac
python task5_frontend/app.py
```

### Issue: FastAPI CORS errors

**Error:**

```
Access to fetch at 'http://localhost:8000/predict' has been blocked by CORS policy
```

**Solution:**

```python
# In api.py, ensure CORS is configured:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Docker build fails

**Error:**

```
ERROR: Could not find a version that satisfies the requirement tensorflow
```

**Solution:**

```dockerfile
# In Dockerfile, use specific base image:
FROM python:3.9-slim

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir tensorflow-cpu==2.13.0
```

### Issue: Model file too large for Git

**Problem:** Cannot push model files (> 100MB)

**Solution:**

```bash
# Option 1: Use Git LFS
git lfs install
git lfs track "*.h5"
git add .gitattributes
git commit -m "Track model files with Git LFS"

# Option 2: Exclude models from Git
# In .gitignore:
models/*.h5

# Option 3: Upload to cloud storage
# Google Drive, Dropbox, AWS S3, etc.
# Add download script:
# python download_models.py
```

### Issue: Heroku deployment exceeds slug size

**Error:**

```
Compiled slug size: 550M is too large (max is 500M)
```

**Solution:**

```bash
# 1. Use tensorflow-cpu instead of tensorflow
# In requirements.txt:
tensorflow-cpu==2.13.0

# 2. Use opencv-python-headless
opencv-python-headless==4.8.0.76

# 3. Remove unnecessary files
# In .slugignore:
*.npy
*.ipynb
task1_problem_definition/
task2_data_preprocessing/processed_data/
*.png
*.jpg
```

---

## Web Interface Issues

### Issue: Image upload not working

**Problem:** Upload button doesn't respond

**Solution:**

```javascript
// In index.html, ensure file input accepts images:
<input type="file" id="fileInput" accept="image/*">

// Check browser console for errors (F12)
// Common issues:
// 1. CORS errors - check API CORS settings
// 2. File size too large - increase limit in Flask/FastAPI
// 3. JavaScript errors - check console
```

### Issue: Predictions are incorrect

**Problem:** Model predicts wrong class

**Solution:**

```python
# 1. Verify preprocessing matches training
# Check image normalization: should be [0, 1]
image_normalized = image.astype(np.float32) / 255.0

# 2. Ensure correct class mapping
class_names = ['With Mask', 'Without Mask', 'Incorrect Mask']
# Should match training order

# 3. Check model is loaded correctly
model = keras.models.load_model('models/best_model.h5')
model.summary()  # Verify architecture

# 4. Test with known images
# Use images from test set with known labels
```

---

## Data Issues

### Issue: Dataset not found

**Error:**

```
FileNotFoundError: archive/images
```

**Solution:**

```bash
# 1. Verify dataset structure
face_mask_detection/
└── archive/
    ├── images/       # Should contain .png files
    └── annotations/  # Should contain .xml files

# 2. Download dataset if missing
# Extract to archive/ folder

# 3. Check file permissions
# Ensure read access to all files
```

### Issue: Corrupted or missing images

**Error:**

```
cv2.error: !_src.empty() in function 'cv::cvtColor'
```

**Solution:**

```python
# Add error handling in preprocessing:
try:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping corrupted image: {img_path}")
        continue
    # Process image...
except Exception as e:
    print(f"Error processing {img_path}: {e}")
    continue
```

---

## General Tips

### Debugging Tips

1. **Enable verbose logging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check data shapes:**

```python
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
```

3. **Verify model summary:**

```python
model.summary()
```

4. **Test with small dataset:**

```python
# Use only 100 samples for quick testing
X_train = X_train[:100]
y_train = y_train[:100]
```

5. **Check GPU usage:**

```bash
# Windows: Task Manager > Performance > GPU
# Linux: nvidia-smi
```

### Performance Tips

1. Use GPU when available
2. Reduce batch size if OOM
3. Use transfer learning instead of training from scratch
4. Cache predictions for repeated requests
5. Use model quantization for deployment

### Best Practices

1. Always use virtual environments
2. Pin dependency versions
3. Add error handling
4. Log important steps
5. Test incrementally
6. Keep backups of trained models
7. Document any customizations

---

## Still Having Issues?

1. **Check logs:** Review error messages carefully
2. **Run verification:** `python verify_setup.py`
3. **Check documentation:** Read task-specific READMEs
4. **Search issues:** Check similar projects on GitHub
5. **Ask for help:** Open an issue with:
   - Error message (full traceback)
   - Environment details (OS, Python version, GPU)
   - Steps to reproduce
   - What you've already tried

---

## Quick Diagnostics

Run this script to check your setup:

```bash
python verify_setup.py
```

This will check:

- Python version
- All dependencies
- GPU availability
- Project structure
- Dataset presence

---

**Most Common Issues:**

1. ✅ Wrong Python version (need 3.9+)
2. ✅ Missing dependencies (run `pip install -r requirements.txt`)
3. ✅ OOM errors (reduce batch_size)
4. ✅ Model not found (run training first)
5. ✅ Port conflicts (use different port or kill process)

---

_This troubleshooting guide covers 90% of common issues. For unique problems, check the specific task's README or open an issue._
