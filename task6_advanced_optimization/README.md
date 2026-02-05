# Task 6: Advanced Optimization

## Transfer Learning with MobileNetV2

### Approach

Transfer learning leverages pre-trained models to achieve better performance with less data and training time.

**MobileNetV2 Architecture:**

- Pre-trained on ImageNet (1.4M images, 1000 classes)
- Lightweight and efficient (3.5M parameters)
- Optimized for mobile and embedded devices
- Excellent feature extraction capabilities

### Implementation Strategy

1. **Base Model**: Load MobileNetV2 pre-trained on ImageNet
2. **Feature Extraction**: Use MobileNetV2 as feature extractor (frozen)
3. **Custom Classifier**: Add custom dense layers for 3-class classification
4. **Fine-Tuning**: (Optional) Unfreeze top layers for domain-specific learning

### Model Architecture

```
MobileNetV2 (Pre-trained, Frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, relu) → Dropout(0.5)
    ↓
BatchNormalization
    ↓
Dense(128, relu) → Dropout(0.3)
    ↓
Dense(3, softmax)
```

### Training Configuration

- **Initial Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 30
- **Optimizer**: Adam
- **Base Model**: Frozen initially
- **Fine-tuning**: Last 20 layers (optional)

### Advantages of Transfer Learning

✅ **Better Accuracy**: Pre-trained features generalize well  
✅ **Faster Training**: Fewer epochs needed  
✅ **Less Data Required**: Works with smaller datasets  
✅ **Reduced Overfitting**: Pre-trained weights act as regularization  
✅ **State-of-the-art Features**: Benefit from ImageNet training

## Model Comparison

### Metrics Compared

1. **Test Accuracy**
2. **Test Loss**
3. **Per-class Precision, Recall, F1-Score**
4. **Training Time**
5. **Model Size**

### Comparison Visualization

- Side-by-side accuracy bars
- Loss comparison
- Detailed classification reports

### Best Model Selection

The model with the highest test accuracy is selected and saved as `best_model.h5` for deployment.

## Files

- `transfer_learning.py`: Main script
- `mobilenetv2_history.json`: Training history
- `comparison_results.json`: Comparison metrics
- `model_comparison.png`: Visual comparison

## Usage

```bash
python task6_advanced_optimization/transfer_learning.py
```

## Expected Results

Transfer learning typically provides:

- **2-5% accuracy improvement** over custom CNN
- **50% faster training** time
- **Better generalization** on unseen data

## Model Deployment

The best performing model is automatically saved as:

```
models/best_model.h5
```

This model should be used for production deployment.
